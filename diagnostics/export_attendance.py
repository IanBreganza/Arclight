import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('attendance.db')
df = pd.read_sql("SELECT * FROM logs ORDER BY timestamp DESC", conn)
conn.close()

filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
df.to_excel(filename, index=False)
print(f"Exported {len(df)} records → {filename}")