import pyarrow.parquet as pq


def read_parquet_file(file_path):
    """
    Reads a Parquet file and returns its contents as a PyArrow Table.

    Parameters:
    file_path (str): The path to the Parquet file.

    Returns:
    pyarrow.Table: The contents of the Parquet file as a PyArrow Table.
    """
    try:
        table = pq.read_table(file_path)
        return table
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return None


pth = "/Users/weronikadomczewska/Documents/magisterskie/semestr3/NLP/0001.parquet"
table = read_parquet_file(pth)
table_pd = table.to_pandas()
print(table_pd)
