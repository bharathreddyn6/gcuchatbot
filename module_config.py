"""
Module Configuration Manager
Manages configurations for all chatbot modules: Library, Study Plan, Academic, Complaints
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import shutil

# Base directory for module data
MODULE_DATA_DIR = Path("module_data")
CONFIG_FILE = MODULE_DATA_DIR / "module_config.json"


class ModuleConfigManager:
    """Centralized configuration manager for all chatbot modules"""
    
    # Module definitions with their configuration schemas
    MODULES = {
        "library": {
            "name": "Library",
            "icon": "📚",
            "description": "Library database for book searches and availability",
            "data_type": "database",
            "connection_fields": {
                "driver": {"type": "select", "options": ["sqlite", "mysql", "postgresql"], "required": True},
                "host": {"type": "text", "required": False, "placeholder": "localhost"},
                "port": {"type": "number", "required": False, "placeholder": "3306"},
                "database": {"type": "text", "required": True, "placeholder": "library_db"},
                "username": {"type": "text", "required": False},
                "password": {"type": "password", "required": False},
                "file_path": {"type": "text", "required": False, "placeholder": "For SQLite: path to .db file"}
            }
        },
        "study_plan": {
            "name": "Study Plan",
            "icon": "📖",
            "description": "Timetable analysis and personalized study plans",
            "data_type": "files",
            "file_types": [".pdf", ".csv", ".png", ".jpg", ".jpeg"],
            "max_file_size_mb": 50,
            "auto_detect_metadata": True  # AI will extract branch/semester from file content
        },
        "academic": {
            "name": "Academic Info",
            "icon": "🎓",
            "description": "Student portal integration for attendance, results, etc.",
            "data_type": "portal",
            "portal_fields": {
                "portal_url": {"type": "text", "required": True, "placeholder": "https://grms.gcu.edu.in"},
                "username": {"type": "text", "required": True},
                "password": {"type": "password", "required": True}
            }
        },
        "complaints": {
            "name": "Complaints",
            "icon": "📝",
            "description": "Student complaint management system",
            "data_type": "database",
            "categories": ["Infrastructure", "Academic", "Hostel", "Transport", "Canteen", "Other"],
            "connection_fields": {
                "driver": {"type": "select", "options": ["sqlite", "mysql", "postgresql"], "required": True},
                "host": {"type": "text", "required": False},
                "port": {"type": "number", "required": False},
                "database": {"type": "text", "required": True},
                "username": {"type": "text", "required": False},
                "password": {"type": "password", "required": False},
                "file_path": {"type": "text", "required": False}
            }
        }
    }
    
    def __init__(self):
        """Initialize configuration manager and create directory structure"""
        self._ensure_directories()
        self._config = self._load_config()
    
    def _ensure_directories(self):
        """Create module data directories if they don't exist"""
        MODULE_DATA_DIR.mkdir(exist_ok=True)
        
        # Create subdirectories for each module
        for module_id in self.MODULES:
            (MODULE_DATA_DIR / module_id).mkdir(exist_ok=True)
        
        # Create study_plan subdirectory for uploaded files
        (MODULE_DATA_DIR / "study_plan" / "uploads").mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load config: {e}")
        
        # Return default config
        return {
            "modules": {
                module_id: {
                    "enabled": True,
                    "config": {},
                    "last_updated": None
                }
                for module_id in self.MODULES
            },
            "global_settings": {
                "admin_password_hash": None,
                "created_at": datetime.now().isoformat()
            }
        }
    
    def _save_config(self):
        """Save configuration to JSON file"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
            raise
    
    def get_module_config(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific module"""
        if module_id not in self.MODULES:
            return None
        return self._config.get("modules", {}).get(module_id, {}).get("config", {})
    
    def set_module_config(self, module_id: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a specific module"""
        if module_id not in self.MODULES:
            return False
        
        if "modules" not in self._config:
            self._config["modules"] = {}
        
        if module_id not in self._config["modules"]:
            self._config["modules"][module_id] = {"enabled": True, "config": {}}
        
        self._config["modules"][module_id]["config"] = config
        self._config["modules"][module_id]["last_updated"] = datetime.now().isoformat()
        
        self._save_config()
        return True
    
    def is_module_enabled(self, module_id: str) -> bool:
        """Check if a module is enabled"""
        return self._config.get("modules", {}).get(module_id, {}).get("enabled", False)
    
    def set_module_enabled(self, module_id: str, enabled: bool) -> bool:
        """Enable or disable a module"""
        if module_id not in self.MODULES:
            return False
        
        self._config["modules"][module_id]["enabled"] = enabled
        self._save_config()
        return True
    
    def get_module_definition(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get module definition (schema)"""
        return self.MODULES.get(module_id)
    
    def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        """Get all module definitions with their current configs"""
        result = {}
        for module_id, definition in self.MODULES.items():
            result[module_id] = {
                **definition,
                "enabled": self.is_module_enabled(module_id),
                "config": self.get_module_config(module_id),
                "last_updated": self._config.get("modules", {}).get(module_id, {}).get("last_updated")
            }
        return result
    
    # ==================== Database Connection Utilities ====================
    
    def test_database_connection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test database connection with given config"""
        driver = config.get("driver", "sqlite")
        
        try:
            if driver == "sqlite":
                import sqlite3
                db_path = config.get("file_path") or config.get("database", "database.db")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                return {"success": True, "message": f"Connected to SQLite: {db_path}"}
            
            elif driver == "mysql":
                import mysql.connector
                conn = mysql.connector.connect(
                    host=config.get("host", "localhost"),
                    port=int(config.get("port", 3306)),
                    database=config.get("database"),
                    user=config.get("username"),
                    password=config.get("password")
                )
                conn.close()
                return {"success": True, "message": f"Connected to MySQL: {config.get('database')}"}
            
            elif driver == "postgresql":
                import psycopg2
                conn = psycopg2.connect(
                    host=config.get("host", "localhost"),
                    port=int(config.get("port", 5432)),
                    database=config.get("database"),
                    user=config.get("username"),
                    password=config.get("password")
                )
                conn.close()
                return {"success": True, "message": f"Connected to PostgreSQL: {config.get('database')}"}
            
            else:
                return {"success": False, "message": f"Unsupported driver: {driver}"}
        
        except ImportError as e:
            return {"success": False, "message": f"Driver not installed: {e}. Install with pip."}
        except Exception as e:
            return {"success": False, "message": f"Connection failed: {str(e)}"}
    
    # ==================== Study Plan File Management ====================
    
    def save_study_plan_file(self, file_path: str, filename: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Save uploaded timetable/study plan file.
        Metadata (branch, semester) will be auto-detected by AI from file content.
        """
        upload_dir = MODULE_DATA_DIR / "study_plan" / "uploads"
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        dest_path = upload_dir / unique_filename
        
        try:
            # Copy file to upload directory
            shutil.copy2(file_path, dest_path)
            
            # Create metadata file
            file_metadata = {
                "original_filename": filename,
                "saved_filename": unique_filename,
                "uploaded_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(dest_path),
                "detected_metadata": metadata or {},  # Will be populated by AI
                "status": "pending_analysis"  # pending_analysis, analyzed, error
            }
            
            metadata_path = dest_path.with_suffix(dest_path.suffix + ".meta.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(file_metadata, f, indent=2)
            
            return {
                "success": True,
                "message": f"File saved: {unique_filename}",
                "file_path": str(dest_path),
                "metadata_path": str(metadata_path)
            }
        
        except Exception as e:
            return {"success": False, "message": f"Failed to save file: {str(e)}"}
    
    def get_study_plan_files(self) -> List[Dict[str, Any]]:
        """Get list of all uploaded study plan files with metadata"""
        upload_dir = MODULE_DATA_DIR / "study_plan" / "uploads"
        files = []
        
        for meta_file in upload_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata["meta_file_path"] = str(meta_file)
                    files.append(metadata)
            except Exception:
                continue
        
        # Sort by upload time, newest first
        files.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return files
    
    def delete_study_plan_file(self, filename: str) -> Dict[str, Any]:
        """Delete a study plan file and its metadata"""
        upload_dir = MODULE_DATA_DIR / "study_plan" / "uploads"
        file_path = upload_dir / filename
        meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        
        try:
            if file_path.exists():
                os.remove(file_path)
            if meta_path.exists():
                os.remove(meta_path)
            return {"success": True, "message": f"Deleted: {filename}"}
        except Exception as e:
            return {"success": False, "message": f"Failed to delete: {str(e)}"}
    
    def update_file_metadata(self, filename: str, detected_metadata: Dict[str, Any]) -> bool:
        """Update detected metadata for a study plan file (called after AI analysis)"""
        upload_dir = MODULE_DATA_DIR / "study_plan" / "uploads"
        meta_path = upload_dir / f"{filename}.meta.json"
        
        if not meta_path.exists():
            return False
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            metadata["detected_metadata"] = detected_metadata
            metadata["status"] = "analyzed"
            metadata["analyzed_at"] = datetime.now().isoformat()
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception:
            return False


# Singleton instance for easy import
config_manager = ModuleConfigManager()


if __name__ == "__main__":
    # Test the configuration manager
    print("Testing ModuleConfigManager...")
    
    mgr = ModuleConfigManager()
    print("\n📦 Available Modules:")
    for module_id, info in mgr.get_all_modules().items():
        print(f"  {info['icon']} {info['name']} - Enabled: {info['enabled']}")
    
    # Test saving library config
    print("\n🔧 Testing Library Configuration...")
    test_config = {
        "driver": "sqlite",
        "file_path": "test_library.db"
    }
    mgr.set_module_config("library", test_config)
    print(f"  Saved config: {mgr.get_module_config('library')}")
    
    # Test connection
    result = mgr.test_database_connection(test_config)
    print(f"  Connection test: {result}")
    
    print("\n✅ ModuleConfigManager test complete!")
