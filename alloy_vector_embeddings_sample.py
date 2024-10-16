########################################################################################################################
"""
Data Processing Script(vectorization) for AlloyDB with Apache Beam for embedding all columns together

This script is designed to read data from an AlloyDB database, process the data using text embeddings
via Vertex AI, and write the transformed data back to AlloyDB. The script utilizes Apache Beam to manage
the data pipeline, which can be executed on Google Cloud Dataflow or locally using the DirectRunner.

The script is intended to be run from the command line and requires various arguments related to AlloyDB
configuration, error handling, table processing, and Beam pipeline options.

Example Usage: python3 -m alloy_vector_embeddings_sample --alloydb_secret_username cwx-alloydb-config --error_bucket_name cwx-poc --error_path mongodb-poc-data/dataflow-error-files --write_to_table_prefix cwx_tealbook_prod_v4_test --read_from_table cwx_tealbook_mongo_prod_data_new_schema_20240823_v2 --read_batch_size 10000 --write_batch_size 10000
"""
########################################################################################################################
import warnings as w
import apache_beam as beam
from apache_beam.runners import DataflowRunner
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions, WorkerOptions, DebugOptions
from apache_beam.ml.transforms.base import MLTransform
from apache_beam.ml.transforms.embeddings.vertex_ai import VertexAITextEmbeddings
from apache_beam.runners.runner import PipelineState
from datetime import datetime
from google.cloud.alloydb.connector import Connector, IPTypes
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import secretmanager, storage
import argparse, json, logging, os, pg8000, shutil, sqlalchemy, tempfile
########################################################################################################################
w.filterwarnings("ignore")
########################################################################################################################
def access_secret_version(project_id, secret_id, version_id = "latest"):
    """
    Accesses the payload for a given secret version in Google Cloud Secret Manager.

    This function retrieves the specified version of a secret stored in Google Cloud Secret Manager.
    If the secret version exists, it returns the decoded payload.

    Args:
        project_id (str): The ID of the Google Cloud project where the secret is stored.
        secret_id (str): The ID of the secret in Secret Manager.
        version_id (str, optional): The version of the secret to access. Defaults to "latest".

    Returns:
        str: The decoded payload of the secret version.
    """
    # Initialize the Secret Manager client
    logging.info("Inside secret manager...")
    client = secretmanager.SecretManagerServiceClient()

    # Construct the resource name for the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    logging.info(f"Accessing secret: {name}")
    # Access the secret version and decode the payload
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")

    # Return the decoded payload
    return payload
########################################################################################################################
def execute_admin_query(admin_query, inst_uri, user, password, db):
    """
    Executes an administrative SQL query on a PostgreSQL database.

    This function establishes a connection to a PostgreSQL database using SQLAlchemy,
    executes the provided administrative query, commits the transaction, and returns
    the result message.

    Args:
        admin_query (str): The SQL query to be executed.
        inst_uri (str): The URI of the PostgreSQL instance.
        user (str): The username for the PostgreSQL database.
        password (str): The password for the PostgreSQL database.
        db (str): The name of the database.

    Returns:
        str: The message returned by the SQL query execution, or a success message 
             if the query does not return any rows.
    """
    try:
        # Debug: Indicate the start of the engine creation process
        print("Before creating engine...")

        # Create SQLAlchemy engine and connector using provided credentials
        engine, connector = create_sqlalchemy_engine(
            inst_uri, user, password, db
        )

        # Debug: Indicate successful engine creation
        print("After creating engine...")
        
        # Debug: Print the administrative query to be executed
        print(admin_query)

        # Establish a connection to the database using the engine
        with engine.connect() as connection:
            # Debug: Indicate that the connection is established
            print("Connection established...")

            # Execute the administrative query
            result = connection.execute(sqlalchemy.text(admin_query))

            # Debug: Indicate before committing the transaction
            print("Before commit...")

            # Commit the transaction to persist changes
            connection.commit()

            # Debug: Indicate after committing the transaction
            print("After commit...")

            # Capture and return the message from the result, or a success message if no rows are returned
            message = result.fetchone()[0] if result.returns_rows else "Table operation successful"
            return message
        
        # Debug: Print the returned message
        print(f"PostgreSQL message: {message}")
    
    except Exception as e:
        # Catch and print any exceptions that occur during the process
        print(f"An unexpected error occurred: {str(e)}")
    
    finally:
        # Ensure the connector is closed after the operation is complete
        connector.close()
########################################################################################################################
def create_sqlalchemy_engine(inst_uri, user, password, db, refresh_strategy = "lazy"):
    """
    Creates a SQLAlchemy engine for connecting to a PostgreSQL database using the Google Cloud SQL Connector.

    This function establishes a connection to a PostgreSQL database using a connection pool created by SQLAlchemy.
    The connection is managed by the Google Cloud SQL Connector, allowing for secure connections using the provided
    instance URI, user credentials, and database name.

    Args:
        inst_uri (str): The URI of the PostgreSQL instance (Cloud SQL instance).
        user (str): The username for the PostgreSQL database.
        password (str): The password for the PostgreSQL database.
        db (str): The name of the database.
        refresh_strategy (str, optional): The strategy for refreshing the connector's credentials. 
                                          Defaults to "lazy".

    Returns:
        Tuple[sqlalchemy.engine.Engine, Connector]: A tuple containing the SQLAlchemy engine and the 
                                                    Google Cloud SQL Connector.
    """
    # Initialize the Google Cloud SQL Connector with the specified refresh strategy
    connector = Connector(refresh_strategy=refresh_strategy)

    def getconn() -> pg8000.dbapi.Connection:
        """
        Establishes and returns a new connection using the Google Cloud SQL Connector.

        This nested function creates a connection to the PostgreSQL instance using the pg8000 driver.

        Returns:
            pg8000.dbapi.Connection: A connection object to the PostgreSQL database.
        """
        conn: pg8000.dbapi.Connection = connector.connect(
            inst_uri,
            "pg8000",
            user=user,
            password=password,
            db=db,
            # Specify the use of public IP to connect to the Cloud SQL instance
            ip_type=IPTypes.PUBLIC,
        )
        return conn

    # Create a SQLAlchemy engine with a connection pool, using the custom connection creator
    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    # Disable the dialect's description encoding (specific to pg8000)
    engine.dialect.description_encoding = None

    # Return both the SQLAlchemy engine and the Cloud SQL Connector
    return engine, connector
########################################################################################################################
class ReadFromAlloyDB(beam.DoFn):
    """
    A custom Apache Beam DoFn class for reading data from an AlloyDB database in chunks.

    This class establishes a connection to an AlloyDB database using SQLAlchemy and processes data 
    in batches. It executes a provided SQL query with pagination using LIMIT and OFFSET, allowing 
    efficient reading of large datasets.

    Attributes:
        inst_uri (str): The URI of the AlloyDB instance.
        user (str): The username for the AlloyDB database.
        password (str): The password for the AlloyDB database.
        db (str): The name of the database.
        query (str): The SQL query to execute for fetching data.
        batch_size (int): The number of records to fetch in each batch. Defaults to 1000.
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine for database connections.
        connector (Connector): The Google Cloud SQL Connector used to establish secure connections.
    """

    def __init__(self, inst_uri, user, password, db, query, batch_size=1000):
        """
        Initializes the ReadFromAlloyDB DoFn.

        Args:
            inst_uri (str): The URI of the AlloyDB instance.
            user (str): The username for the AlloyDB database.
            password (str): The password for the AlloyDB database.
            db (str): The name of the database.
            query (str): The SQL query to execute for fetching data.
            batch_size (int, optional): The number of records to fetch in each batch. Defaults to 1000.
        """
        self.inst_uri = inst_uri
        self.user = user
        self.password = password
        self.db = db
        self.batch_size = batch_size
        self.engine = None
        self.connector = None
        self.query = query

    def setup(self):
        """Create the SQLAlchemy engine and connector in the setup phase."""
        self.engine, self.connector = create_sqlalchemy_engine(
            self.inst_uri, self.user, self.password, self.db
        )

    def process(self, element):
        """
        Fetch a specific chunk of data from AlloyDB based on the element received.

        Args:
            element (int): The split index used to determine the OFFSET for the SQL query.
        
        Yields:
            dict: A dictionary representing a row of data fetched from the database.
        """
        logging.info(f"Element: {element}")
        split_index = element

        # Calculate the OFFSET based on the current split index and batch size
        offset = int(split_index) * int(self.batch_size)
        logging.info(f"After setting offset: {offset}")
        
        with self.engine.connect() as conn:
            try:
                # Debug: Print the current offset and split index
                print(f"Offset: {offset}")
                print(f"Split Index: {split_index}")

                # Prepare the SQL query with LIMIT and OFFSET for pagination
                sql_statement = sqlalchemy.text(f"{self.query} LIMIT :limit OFFSET :offset")
                stmt = sql_statement.bindparams(limit=self.batch_size, offset=offset)

                # Compile the SQL statement for logging and debugging
                compiled_stmt = stmt.compile(self.engine)
                logging.info(f"SQL Statement: {str(compiled_stmt)}")
                logging.info(f"Limit: {str(self.batch_size)}")
                logging.info(f"Offset: {str(offset)}")

                # Execute the SQL query and fetch the result rows
                result = conn.execute(sql_statement, {'limit': self.batch_size, 'offset': offset})
                rows = result.fetchall()

                # Yield each row as a dictionary
                for row in rows:
                    yield dict(row._mapping)
            except Exception as e:
                # Handle any exceptions that occur during the database query
                print(f"Error reading from AlloyDB: {e}")
        return

    def teardown(self):
        """Close the connector and dispose of the engine when the bundle is finished."""
        self.connector.close()
        self.engine.dispose()
########################################################################################################################
class WriteToAlloyDB(beam.DoFn):
    """
    A custom Apache Beam DoFn class for writing data to an AlloyDB database in batches.

    This class handles buffering of elements, batch insertion into the AlloyDB database, and error handling.
    It also supports writing error records to Google Cloud Storage (GCS) in case of failures during the insertion process.

    Attributes:
        inst_uri (str): The URI of the AlloyDB instance.
        user (str): The username for the AlloyDB database.
        password (str): The password for the AlloyDB database.
        db (str): The name of the database.
        table_name (str): The name of the table where data will be inserted.
        batch_size (int): The number of records to insert in each batch.
        job_name (str): The name of the Dataflow job, used for error logging.
        error_bucket_name (str): The name of the GCS bucket where error records will be stored.
        error_path (str): The GCS path where error records will be saved.
        insert_line_1_str (str): The part of the SQL insert statement before the VALUES clause.
        insert_line_2_str (str): The part of the SQL insert statement inside the VALUES clause.
    """

    def __init__(self, insert_line_1_str, insert_line_2_str, job_name, error_bucket_name, error_path, inst_uri, user, password, db, table_name, batch_size):
        """
        Initializes the WriteToAlloyDB DoFn.

        Args:
            column_name (str): The name of the column that is the focus of the insertion process.
            insert_line_1_str (str): The part of the SQL insert statement before the VALUES clause.
            insert_line_2_str (str): The part of the SQL insert statement inside the VALUES clause.
            job_name (str): The name of the Dataflow job, used for error logging.
            error_bucket_name (str): The name of the GCS bucket where error records will be stored.
            error_path (str): The GCS path where error records will be saved.
            inst_uri (str): The URI of the AlloyDB instance.
            user (str): The username for the AlloyDB database.
            password (str): The password for the AlloyDB database.
            db (str): The name of the database.
            table_name (str): The name of the table where data will be inserted.
            batch_size (int): The number of records to insert in each batch.
        """
        self.inst_uri = inst_uri
        self.user = user
        self.password = password
        self.db = db
        self.batch_size = int(batch_size)
        self.engine = None
        self.connector = None
        self.buffer = []
        self.job_name = job_name
        self.error_bucket_name = error_bucket_name
        self.error_path = error_path
        self.table_name = table_name
        self.insert_line_1_str = insert_line_1_str
        self.insert_line_2_str = insert_line_2_str 

    def setup(self):
        """Create the SQLAlchemy engine and connector in the setup phase."""
        self.engine, self.connector = create_sqlalchemy_engine(
            self.inst_uri, self.user, self.password, self.db
        )

    def process(self, element):
        """
        Buffer elements and write to AlloyDB in batches.

        Args:
            element (dict or list): The element(s) to be processed and inserted into the database.
        """
        if isinstance(element, dict):
            # If the element is a dictionary, add it directly to the buffer
            self.buffer.append(element)
        elif isinstance(element, list):
            # If the element is a list, iterate through and add each dictionary to the buffer
            for item in element:
                if isinstance(item, dict):
                    self.buffer.append(item)
                else:
                    print(f"Warning: List item is not a dictionary: {item}")
        else:
            print(f"Warning: Element is neither a dictionary nor a list: {element}")

        # If the buffer size reaches the batch size, flush the buffer to the database
        if len(self.buffer) >= int(self.batch_size):
            self.flush_buffer()

    def flush_buffer(self):
        """
        Insert buffered elements into the database.

        This method constructs an SQL insert statement and executes it in a batch. If the batch insert fails,
        it falls back to inserting records individually.
        """
        if not self.buffer:
            return

        def format_vector(vector):
            """Format the vector for insertion into the database."""
            logging.info(f"Inside format vector...")
            if vector is not None:
                return f"[{', '.join(map(str, vector))}]"
            return None

        def is_valid_json(data):
            try:
                # Attempt to parse the data as JSON
                parsed_data = json.loads(data)
                # If it’s valid JSON, you can re-dump it or process it further
                json_string = json.dumps(parsed_data)
            except TypeError as e:
                #print(f"Error:{e}")
                json_string = data
            return json_string

        with self.engine.connect() as conn:
            transaction = conn.begin()
            try:
                # Construct the SQL insert statement
                sql_statement = sqlalchemy.text(
                    f"INSERT INTO {self.table_name} (col1, col2) " # Column List
                    f"VALUES (:col1, :col2)" # Column List
                )

                # Prepare the list of parameters for the batch insert
                params_list = [{    'col1': str(record.get('col1', None)),
                                    'col2': record.get('col2', None),
                                    'col1_embedding_vector': format_vector(record.get(f'col1_embedding', None)),
                                    'col2_embedding_vector': format_vector(record.get(f'col2_embedding', None)), 
                                } for record in self.buffer]

                # Execute the batch insert
                conn.execute(sql_statement, params_list)
                transaction.commit()  # Commit the transaction
                self.buffer = []  # Clear the buffer
            except Exception as batch_exception:
                # Handle batch insert failure by rolling back and trying individual inserts
                print(f"Batch insert error: {batch_exception}")
                transaction.rollback()
                self.handle_individual_inserts(conn, sql_statement)

    def handle_individual_inserts(self, conn, sql_statement):
        """
        Handle individual inserts in case of a batch insert failure.

        Args:
            conn (sqlalchemy.engine.Connection): The active database connection.
            sql_statement (sqlalchemy.sql.elements.TextClause): The prepared SQL statement.
        """
        def format_vector(vector):
            """Format the vector for insertion into the database."""
            logging.info(f"Inside format vector...")
            if vector is not None:
                return f"[{', '.join(map(str, vector))}]"
            return None

        def is_valid_json(data):
            try:
                # Attempt to parse the data as JSON
                parsed_data = json.loads(data)
                # If it’s valid JSON, you can re-dump it or process it further
                json_string = json.dumps(parsed_data)
            except TypeError as e:
                #print(f"Error:{e}")
                json_string = data
            return json_string

        for record in self.buffer:
            try:
                with conn.begin() as individual_transaction:
                    # Prepare the parameters for the individual insert
                    params = {      'col1': str(record.get('col1', None)),
                                    'col2': record.get('col2', None),
                                    'col1_embedding_vector': format_vector(record.get(f'col1_embedding', None)),
                                    'col2_embedding_vector': format_vector(record.get(f'col2_embedding', None)),
                            }
                    # Execute the individual insert
                    conn.execute(sql_statement, [params])
                    individual_transaction.commit()
            except Exception as record_exception:
                # Handle individual insert failure by logging the error and writing to GCS
                individual_transaction.rollback()
                self.write_errors_to_gcs([record], record_exception)
                print(f"Error writing individual record to AlloyDB: {record_exception}")

    def write_errors_to_gcs(self, error_records, exception):
        """
        Write errored out records to a GCS file.

        Args:
            error_records (list): The list of records that failed to insert.
            exception (Exception): The exception that caused the failure.
        """
        # Access the GCS bucket
        bucket = storage.Client().bucket(self.error_bucket_name)
        # Construct the error file path with a timestamp
        error_filename = f"{self.error_path}/{self.job_name}/errors-of-job-{self.job_name}-at-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.json"
        blob = bucket.blob(error_filename)

        # Prepare the error data to be written to the file
        error_data = {
            "error_message": str(exception),
            "records": error_records
        }

        # Upload the error data to GCS as a JSON file
        blob.upload_from_string(
            data=json.dumps(error_data, default=str),
            content_type="application/json"
        )
        print(f"Error records written to GCS: gs://{self.error_bucket_name}/{error_filename}")

    def finish_bundle(self):
        """Flush any remaining elements in the buffer at the end of the bundle."""
        self.flush_buffer()

    def teardown(self):
        """Close the connector and dispose of the engine when the bundle is finished."""
        self.connector.close()
        self.engine.dispose()
########################################################################################################################
def run(alloydb_secret_username, error_bucket_name, error_path, write_to_table_prefix, read_from_table, read_batch_size, write_batch_size):
    """
    Runs a Dataflow pipeline that reads data from AlloyDB, performs text embedding using Vertex AI,
    and writes the transformed data back to AlloyDB.

    This function sets up the necessary configurations, prepares the SQL queries, and manages the Dataflow 
    pipeline for processing data from a specified column in AlloyDB. The pipeline includes reading data, 
    performing embeddings, and writing the results back to a new table in AlloyDB.

    Args:
        alloydb_secret_username (str): The Secret Manager key for accessing AlloyDB credentials.
        error_bucket_name (str): The name of the GCS bucket where error records will be stored.
        error_path (str): The GCS path where error records will be saved.
        write_to_table_prefix (str): The name of the table where the processed data will be written.
        read_from_table (str): The name of the table to read from
        read_batch_size (int): Read batch size
        write_batch_size (int): Write batch size

    """
    # Get the current timestamp for job naming and logging
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Define a unique job name using the current timestamp and column name
    job_name = f"alloy-vector-embeddings-v23-prod-" + timestamp_str
    
    # Initialize Apache Beam pipeline options
    op = PipelineOptions()
    op.view_as(SetupOptions).requirements_file = 'requirements.txt'
    op.view_as(SetupOptions).save_main_session = True
    op.view_as(GoogleCloudOptions).project = 'cwx-poc'
    op.view_as(GoogleCloudOptions).job_name = job_name
    op.view_as(GoogleCloudOptions).temp_location = 'gs://cwx-poc/mongodb-poc-data/dataflow-temp-files'
    op.view_as(GoogleCloudOptions).staging_location = 'gs://cwx-poc/mongodb-poc-data/dataflow-temp-files/staging'
    op.view_as(GoogleCloudOptions).region = 'us-central1'
    # Uncomment the following line to run locally instead of on Dataflow
    # op.view_as(StandardOptions).runner = 'DirectRunner'
    op.view_as(StandardOptions).runner = 'DataflowRunner'
    op.view_as(WorkerOptions).num_workers = 4
    op.view_as(WorkerOptions).autoscaling_algorithm = 'NONE'
    op.view_as(WorkerOptions).machine_type = 'n1-standard-16'
    
    # Retrieve AlloyDB credentials from Secret Manager
    project_id = "cwx-poc"
    secret_id_alloy = alloydb_secret_username
    version_id = "latest"  # Specify the version of the secret to retrieve
    secret_value_alloy = access_secret_version(project_id, secret_id_alloy, version_id)
    
    # Load the AlloyDB configuration from the retrieved secret
    alloydb_config = json.loads(secret_value_alloy)

    # Define the name of the test table to create
    write_to_table =  write_to_table_prefix + "_test"
    
    # Drop the existing test table if it exists and create a new one
    execute_admin_query(
        admin_query=f"""DROP TABLE IF EXISTS public.{write_to_table}; 
                        CREATE TABLE public.{write_to_table} (LIKE public.{read_from_table} INCLUDING ALL);""", 
        inst_uri=alloydb_config["inst_uri"], 
        user=alloydb_config["user"], 
        password=alloydb_config["password"], 
        db=alloydb_config["db"]
    )
    
    # Prepare SQL fragments for adding columns, selecting columns, and inserting data
    keys_to_check = ['col1', 'col2'] # List of columns to be embedded: comma-seperated
    read_query_list = []
    add_column = []
    select_column = []
    columns_to_embed = []
    insert_line_1 = []
    insert_line_2 = []
    
    for key in keys_to_check:
        add_column.append(f"ADD COLUMN {key}_embedding_vector vector(768)") 
        select_column.append(f"{key} AS {key}_embedding")
        columns_to_embed.append(f"{key}_embedding")
        read_query_list.append(f"COALESCE(NULLIF({key}, ''), ' ') AS {key}_embedding")
        insert_line_1.append(f"{key}_embedding_vector")
        insert_line_2.append(f"CASE WHEN :{key} NOT IN ('None', '') THEN CAST(:{key}_embedding_vector AS vector(768)) ELSE NULL END")
    
    # Convert lists to strings for use in SQL queries
    add_column_str = ", ".join(add_column)
    select_column_str = ", ".join(select_column)
    read_query_list_str = ",".join(read_query_list)
    insert_line_1_str = ",".join(insert_line_1)
    insert_line_2_str = ",".join(insert_line_2)
    
    # Alter the test table to add the new vector columns
    execute_admin_query(
        admin_query=f"ALTER TABLE public.{write_to_table} {add_column_str}", 
        inst_uri=alloydb_config["inst_uri"], 
        user=alloydb_config["user"], 
        password=alloydb_config["password"], 
        db=alloydb_config["db"]
    )
    
    # Construct the read query to retrieve data for embedding
    read_query = f"""
        SELECT *, {read_query_list_str} 
        FROM public.{read_from_table}
    """
    print(read_query)

    # Remove any existing temporary files from a previous run
    if 'artifact_location' in locals() or 'artifact_location' in globals():
        if os.path.exists(artifact_location):
            print("Removing old artifact location...")
            shutil.rmtree(artifact_location)

    # Create a new temporary directory for storing artifacts
    artifact_location = tempfile.mkdtemp(prefix='vertex_ai')

    # Specify the Vertex AI model for text embedding
    text_embedding_model_name = 'text-embedding-004'

    # Calculate the number of splits based on the number of rows in the source table
    num_rows = execute_admin_query(
        admin_query = f"""
            SELECT COUNT(1) 
            FROM public.{read_from_table}
        """, 
        inst_uri = alloydb_config["inst_uri"], 
        user = alloydb_config["user"], 
        password = alloydb_config["password"], 
        db = alloydb_config["db"]
    )
    num_splits = (num_rows // 10000) + 1
    print(f"Total Rows: {num_rows}")
    print(f"Number of Splits: {num_splits}")

    # Build and run the Apache Beam pipeline
    with beam.Pipeline(options = op) as pipeline:
        (pipeline
         | 'Create Splits' >> beam.Create(range(num_splits))
         | 'Read from AlloyDB' >> beam.ParDo(
                ReadFromAlloyDB(
                    inst_uri = alloydb_config["inst_uri"], 
                    user = alloydb_config["user"], 
                    password = alloydb_config["password"], 
                    db = alloydb_config["db"], 
                    query = f"{read_query}", 
                    batch_size = read_batch_size
                )
            )
         | 'MLTransformAS' >> MLTransform(
                write_artifact_location = artifact_location
            ).with_transform(
                VertexAITextEmbeddings(
                    model_name = text_embedding_model_name, 
                    columns = columns_to_embed, 
                    project = project_id
                )
            )
         # Uncomment the following line to debug output
         # | beam.Map(print)
         | 'Write to AlloyDB' >> beam.ParDo(
                WriteToAlloyDB(
                    insert_line_1_str,
                    insert_line_2_str,
                    job_name, 
                    error_bucket_name, 
                    error_path, 
                    inst_uri = alloydb_config["inst_uri"], 
                    user = alloydb_config["user"], 
                    password = alloydb_config["password"], 
                    db = alloydb_config["db"], 
                    table_name = write_to_table, 
                    batch_size = write_batch_size
                )
            )
        )
########################################################################################################################
if __name__ == '__main__':
    """
    Main entry point for the script.

    This script initializes the logging, parses command-line arguments, and runs the main processing function.
    The script is designed to be executed directly and expects various arguments related to AlloyDB configuration,
    error handling, and table processing.

    Command-line Arguments:
        --alloydb_secret_username (str): The Secret Manager key for accessing AlloyDB credentials.
        --error_bucket_name (str): The name of the GCS bucket where error records will be stored.
        --error_path (str): The GCS path where error records will be saved.
        --write_to_table_prefix (str): The prefix of the table to which data will be written.
        --read_from_table (str): The name of the table from which data will be read.
        --read_batch_size (int): The batch size for reading data from the source table.
        --write_batch_size (int): The batch size for writing data to the destination table.
    """
    
    # Set up logging configuration to log informational messages
    logging.getLogger().setLevel(logging.INFO)
    
    # Initialize argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    
    # Define the required command-line arguments with descriptions
    parser.add_argument("--alloydb_secret_username", required = True, help = "AlloyDB Secret Username")
    parser.add_argument("--error_bucket_name", required = True, help = "GCS bucket name for storing error records")
    parser.add_argument("--error_path", required = True, help = "GCS path for storing error records")
    parser.add_argument("--write_to_table_prefix", required = True, help = "Prefix of the table to write the data to")
    parser.add_argument("--read_from_table", required = True, help = "Name of the table to read data from")
    parser.add_argument("--read_batch_size", required = True, help = "Batch size for reading data")
    parser.add_argument("--write_batch_size", required = True, help = "Batch size for writing data")
    # parser.add_argument("--column_name", required = True, help = "Name of the column to process")

    # Parse known arguments, separating them from any additional pipeline-specific arguments
    known_args, pipeline_args = parser.parse_known_args()

    # Run the main processing function with the parsed arguments
    run(
        known_args.alloydb_secret_username,
        known_args.error_bucket_name,
        known_args.error_path,
        known_args.write_to_table_prefix,
        known_args.read_from_table,
        known_args.read_batch_size,
        known_args.write_batch_size,
        # known_args.column_name
    )
########################################################################################################################
# Execution command for the script: 
# python3 -m alloy_vector_embeddings_sample --alloydb_secret_username cwx-alloydb-config --error_bucket_name cwx-poc --error_path mongodb-poc-data/dataflow-error-files --write_to_table_prefix cwx_tealbook_prod_v4_test --read_from_table cwx_tealbook_mongo_prod_data_new_schema_20240823_v2 --read_batch_size 10000 --write_batch_size 10000
########################################################################################################################