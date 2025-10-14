"""
Suricata integration for the IDS system
Handles Suricata process management and alert processing
"""
import os
import json
import subprocess
import threading
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger('nids_improved')

class SuricataManager:
    def __init__(self, 
                 suricata_bin: str = 'suricata',
                 config_path: str = None,
                 rules_dir: str = None,
                 log_dir: str = 'logs/suricata',
                 interface: str = 'eth0'):
        """
        Initialize the Suricata manager
        
        Args:
            suricata_bin: Path to suricata binary
            config_path: Path to suricata.yaml
            rules_dir: Directory containing Suricata rules
            log_dir: Directory for Suricata logs
            interface: Network interface to monitor
        """
        self.suricata_bin = suricata_bin
        self.interface = interface
        self.process = None
        self.running = False
        
        # Set default paths if not provided
        self.config_path = config_path or self._find_default_config()
        self.rules_dir = rules_dir or self._find_default_rules_dir()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert callback function
        self.alert_callback = None
        
        # Alert file paths
        self.alert_json = self.log_dir / 'eve.json'
        
        logger.info(f"Initialized SuricataManager with config: {self.config_path}")
        logger.info(f"Rules directory: {self.rules_dir}")
        logger.info(f"Log directory: {self.log_dir}")
    
    def _find_default_config(self) -> Optional[Path]:
        """Try to find default suricata.yaml in common locations"""
        possible_paths = [
            '/etc/suricata/suricata.yaml',
            '/usr/local/etc/suricata/suricata.yaml',
            'C:\\Program Files\\Suricata\\suricata.yaml',
            'C:\\Program Files (x86)\\Suricata\\suricata.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return Path(path)
        return None
    
    def _find_default_rules_dir(self) -> Optional[Path]:
        """Try to find default rules directory"""
        possible_paths = [
            '/etc/suricata/rules',
            '/usr/local/var/lib/suricata/rules',
            'C:\\Program Files\\Suricata\\rules',
            'C:\\Program Files (x86)\\Suricata\\rules'
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                return Path(path)
        return None
    
    def set_alert_callback(self, callback: Callable[[Dict], None]):
        """Set the callback function for handling alerts"""
        self.alert_callback = callback
    
    def _read_alerts(self):
        """Continuously read alerts from the JSON log file"""
        if not self.alert_json.exists():
            logger.warning(f"Alert file {self.alert_json} does not exist yet")
            return
            
        # Read existing content to skip
        if self.alert_json.exists():
            with open(self.alert_json, 'r', encoding='utf-8') as f:
                for _ in f:
                    pass  # Skip existing content
        
        # Continuously read new alerts
        while self.running:
            try:
                with open(self.alert_json, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            alert = json.loads(line)
                            if alert.get('event_type') == 'alert' and self.alert_callback:
                                self.alert_callback(alert)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse alert: {line}")
                            continue
            except Exception as e:
                logger.error(f"Error reading alerts: {e}")
                
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    def start(self):
        """Start Suricata in live capture mode"""
        if self.running:
            logger.warning("Suricata is already running")
            return True
            
        # Check if Suricata binary exists
        if not os.path.isfile(self.suricata_bin):
            logger.warning(f"Suricata binary not found at {self.suricata_bin}")
            return False
            
        # Check if config file exists
        if not os.path.isfile(self.config_path):
            logger.warning(f"Suricata config file not found at {self.config_path}")
            return False
            
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the Suricata command with minimal required options
        cmd = [
            self.suricata_bin,
            '-c', str(self.config_path),
            '-i', self.interface,
            '--set', f'default-rule-path={self.rules_dir}',
            '-l', str(self.log_dir),
            '--pidfile', str(self.log_dir / 'suricata.pid'),
            '--af-packet',
            '-v',
            '--no-random',
            '--set', 'logging.outputs.0.eve-log.types=[alert]',
            '--set', 'detection.mpm-algo=ac-ks',
            '--runmode', 'workers'  # Use workers runmode for better performance
        ]
        
        try:
            logger.info(f"Starting Suricata with command: {' '.join(cmd)}")
            
            # Use a temporary file for stderr to capture any startup errors
            with open(str(self.log_dir / 'suricata_stderr.log'), 'w') as stderr_file:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True
                )
            
            # Wait a bit longer for Suricata to start
            time.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is not None:
                # Read the error log
                try:
                    with open(str(self.log_dir / 'suricata_stderr.log'), 'r') as f:
                        error_output = f.read()
                    logger.error(f"Suricata failed to start. Error output:\n{error_output}")
                except Exception as e:
                    logger.error(f"Failed to read Suricata error log: {e}")
                return False
            
            self.running = True
            
            # Start alert reader thread
            self.alert_thread = threading.Thread(
                target=self._read_alerts,
                daemon=True,
                name="SuricataAlertReader"
            )
            self.alert_thread.start()
            
            logger.info("Suricata started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Suricata: {e}", exc_info=True)
            self.running = False
            if hasattr(self, 'process') and self.process:
                try:
                    self.process.terminate()
                except:
                    pass
            return False
    
    def stop(self):
        """Stop Suricata"""
        if not self.running:
            return
            
        self.running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Suricata did not terminate gracefully, forcing kill")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping Suricata: {e}")
        
        if hasattr(self, 'alert_thread') and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=2)
        
        logger.info("Suricata stopped")
    
    def reload_rules(self):
        """Reload Suricata rules"""
        if not self.process:
            logger.warning("Cannot reload rules: Suricata is not running")
            return False
            
        try:
            # Send SIGHUP to reload rules
            self.process.send_signal(1)  # SIGHUP
            logger.info("Reloaded Suricata rules")
            return True
        except Exception as e:
            logger.error(f"Failed to reload Suricata rules: {e}")
            return False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    def alert_callback(alert):
        print(f"[ALERT] {alert.get('signature', 'Unknown signature')} - {alert.get('src_ip')} -> {alert.get('dest_ip')}")
    
    with SuricataManager(interface='eth0') as suricata:
        suricata.set_alert_callback(alert_callback)
        print("Suricata is running. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping Suricata...")
