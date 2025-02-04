"""Data Persistence Module

This module provides functionality for persisting and retrieving data
in embodied AI applications.
"""

from typing import Any, Dict, Optional
import json
import os
import sqlite3
import numpy as np
from datetime import datetime

class DataPersistence:
    """Handles data persistence operations for EmbodyHub.
    
    This class provides methods for storing and retrieving multimodal data,
    including support for different storage backends and data formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the data persistence manager.
        
        Args:
            config: Optional configuration dictionary for storage settings.
        """
        self.config = config or {}
        self.storage_path = self.config.get('storage_path', 'data')
        self.db_path = os.path.join(self.storage_path, 'embodyhub.db')
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage backend and create necessary structures."""
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize SQLite database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables for different data types
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_streams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stream_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id INTEGER,
                    data BLOB NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (stream_id) REFERENCES data_streams(id)
                )
            """)
            
            conn.commit()
    
    def save_data(self, stream_name: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """Save data to persistent storage.
        
        Args:
            stream_name: Name of the data stream.
            data: Data to be stored.
            metadata: Optional metadata associated with the data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get or create stream record
            cursor.execute(
                "SELECT id FROM data_streams WHERE stream_name = ?",
                (stream_name,)
            )
            result = cursor.fetchone()
            
            if result is None:
                # Create new stream record
                data_type = self._detect_data_type(data)
                cursor.execute(
                    "INSERT INTO data_streams (stream_name, data_type) VALUES (?, ?)",
                    (stream_name, data_type)
                )
                stream_id = cursor.lastrowid
            else:
                stream_id = result[0]
            
            # Save the data
            serialized_data = self._serialize_data(data)
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute(
                "INSERT INTO stream_data (stream_id, data, metadata) VALUES (?, ?, ?)",
                (stream_id, serialized_data, metadata_json)
            )
            
            conn.commit()
    
    def load_data(self, stream_name: str, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> list:
        """Load data from persistent storage.
        
        Args:
            stream_name: Name of the data stream to load.
            start_time: Optional start time for data retrieval.
            end_time: Optional end time for data retrieval.
            
        Returns:
            List of tuples containing (data, metadata, timestamp).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT sd.data, sd.metadata, sd.timestamp
                FROM stream_data sd
                JOIN data_streams ds ON sd.stream_id = ds.id
                WHERE ds.stream_name = ?
            """
            params = [stream_name]
            
            if start_time:
                query += " AND sd.timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                query += " AND sd.timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY sd.timestamp"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [
                (
                    self._deserialize_data(row[0]),
                    json.loads(row[1]) if row[1] else None,
                    datetime.fromisoformat(row[2])
                )
                for row in results
            ]
    
    def retrieve_data(self, stream_name: str, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retrieve data from a specific stream within a time range.
        
        Args:
            stream_name: Name of the data stream.
            start_time: Optional start time for data retrieval.
            end_time: Optional end time for data retrieval.
            
        Returns:
            List of data entries with their metadata.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT sd.data, sd.metadata, sd.timestamp
                FROM stream_data sd
                JOIN data_streams ds ON sd.stream_id = ds.id
                WHERE ds.stream_name = ?
            """
            params = [stream_name]
            
            if start_time:
                query += " AND sd.timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                query += " AND sd.timestamp <= ?"
                params.append(end_time.isoformat())
                
            query += " ORDER BY sd.timestamp ASC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [
                {
                    'data': self._deserialize_data(row[0]),
                    'metadata': json.loads(row[1]) if row[1] else None,
                    'timestamp': row[2]
                }
                for row in results
            ]
    
    def get_stream_statistics(self, stream_name: str) -> Dict[str, Any]:
        """Get statistical information about a data stream.
        
        Args:
            stream_name: Name of the data stream.
            
        Returns:
            Dictionary containing stream statistics.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as entry_count,
                    MIN(timestamp) as first_entry,
                    MAX(timestamp) as last_entry
                FROM stream_data sd
                JOIN data_streams ds ON sd.stream_id = ds.id
                WHERE ds.stream_name = ?
            """, (stream_name,))
            
            count, first, last = cursor.fetchone()
            
            return {
                'entry_count': count,
                'first_entry': first,
                'last_entry': last,
                'stream_name': stream_name
            }
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage."""
        if isinstance(data, (dict, list)):
            return json.dumps(data).encode('utf-8')
        elif isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            return str(data).encode('utf-8')
    
    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            return json.loads(data_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            try:
                return np.frombuffer(data_bytes)
            except:
                return data_bytes.decode('utf-8')
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data for storage optimization."""
        if isinstance(data, dict):
            return 'json'
        elif isinstance(data, np.ndarray):
            return 'numpy'
        elif isinstance(data, (list, tuple)):
            return 'array'
        else:
            return 'text'
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data for appropriate storage handling.
        
        Args:
            data: Data to analyze.
            
        Returns:
            String indicating the data type.
        """
        if isinstance(data, np.ndarray):
            return f'numpy.{data.dtype}'
        return type(data).__name__
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage.
        
        Args:
            data: Data to serialize.
            
        Returns:
            Serialized data as bytes.
        """
        if isinstance(data, np.ndarray):
            return data.tobytes()
        return json.dumps(data).encode('utf-8')
    
    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize data from storage.
        
        Args:
            data_bytes: Serialized data to restore.
            
        Returns:
            Deserialized data.
        """
        try:
            return json.loads(data_bytes)
        except:
            # Assume numpy array if JSON deserialization fails
            return np.frombuffer(data_bytes)
    
    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the current data.
        
        Args:
            backup_path: Optional path for the backup file.
            
        Returns:
            Path to the created backup file.
        """
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.storage_path, f'backup_{timestamp}.db')
        
        with sqlite3.connect(self.db_path) as source:
            backup = sqlite3.connect(backup_path)
            source.backup(backup)
            backup.close()
        
        return backup_path
    
    def restore(self, backup_path: str) -> None:
        """Restore data from a backup.
        
        Args:
            backup_path: Path to the backup file to restore from.
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Create a temporary database for validation
        temp_path = backup_path + '.temp'
        try:
            with sqlite3.connect(backup_path) as source:
                temp = sqlite3.connect(temp_path)
                source.backup(temp)
                temp.close()
            
            # If validation successful, restore to main database
            with sqlite3.connect(backup_path) as source:
                target = sqlite3.connect(self.db_path)
                source.backup(target)
                target.close()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)