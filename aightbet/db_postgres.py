import pandas as pd
import psycopg
from . import config

def get_connection():
    """Establishes a connection to the PostgreSQL database."""
    return psycopg.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )

def fetch_quotes(start_date, end_date):
    """Fetches quote data within a date range."""
    query = """
        SELECT datetime, tickersymbol, price
        FROM "quote"."matched"
        WHERE datetime >= %s and datetime < %s
        and tickersymbol LIKE 'VN30F2%%'
        ORDER BY datetime
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (start_date, end_date))
            quotes = pd.DataFrame(cur.fetchall(), columns=['datetime', 'tickersymbol', 'price'])
            
    # Process types
    quotes['datetime'] = pd.to_datetime(quotes['datetime'])
    quotes.set_index('datetime', inplace=True)
    quotes['price'] = quotes['price'].astype(float)
    
    return quotes

def fetch_volume(start_date, end_date):
    """Fetches volume data within a date range."""
    query = """
        SELECT datetime, tickersymbol, quantity
        FROM "quote"."matchedvolume"
        WHERE datetime >= %s and datetime < %s
        and tickersymbol LIKE 'VN30F2%%'
        ORDER BY datetime
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (start_date, end_date))
            volume = pd.DataFrame(cur.fetchall(), columns=['datetime', 'tickersymbol', 'quantity'])
            
    # Process types
    volume['datetime'] = pd.to_datetime(volume['datetime'])
    volume.set_index('datetime', inplace=True)
    volume['quantity'] = volume['quantity'].astype(float)
    
    return volume
