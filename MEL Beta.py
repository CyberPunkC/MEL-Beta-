#!/usr/bin/env python3
"""
MEL - Assistant IA Vocal Intelligent
Auteur: Votre Nom
Version: 1.0.0
Description: Assistant vocal avec accès système complet
"""

import os
import sys
import json
import time
import datetime
import threading
import queue
import subprocess
import webbrowser
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Modules vocaux
import speech_recognition as sr
import pyttsx3

# Modules NLP
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Modules système
import psutil
import pyautogui
import keyboard
import pyperclip
import requests

# Modules ML et utilitaires
import numpy as np
import pandas as pd
from loguru import logger
import yaml
import schedule

# Configuration des logs
logger.add("mel_logs_{time}.log", rotation="1 week")

# ==========================
# Configuration et Constants
# ==========================

@dataclass
class Config:
    """Configuration centrale de MEL"""
    # Vocal
    LANGUAGE: str = "fr-FR"
    WAKE_WORD: str = "mel"
    VOICE_RATE: int = 180
    VOICE_VOLUME: float = 0.9
    
    # Système
    CONFIDENCE_THRESHOLD: float = 0.7
    TIMEOUT_LISTENING: int = 5
    MAX_RETRIES: int = 3
    
    # Chemins
    BASE_DIR: Path = Path.home() / ".mel"
    COMMANDS_FILE: Path = BASE_DIR / "commands.json"
    HISTORY_FILE: Path = BASE_DIR / "history.json"
    SETTINGS_FILE: Path = BASE_DIR / "settings.yaml"
    
    # Sécurité
    SAFE_MODE: bool = True
    REQUIRE_CONFIRMATION: List[str] = ["shutdown", "delete", "format", "install"]

class CommandType(Enum):
    """Types de commandes"""
    SYSTEM = "system"
    APPLICATION = "application"
    FILE = "file"
    WEB = "web"
    INFORMATION = "information"
    AUTOMATION = "automation"
    SETTINGS = "settings"

# ==========================
# Module Principal MEL
# ==========================

class MEL:
    """Classe principale de l'assistant MEL"""
    
    def __init__(self):
        logger.info("Initialisation de MEL...")
        
        # Configuration
        self.config = Config()
        self._setup_directories()
        self._load_settings()
        
        # Modules
        self.voice_engine = VoiceEngine(self.config)
        self.nlp_processor = NLPProcessor(self.config)
        self.command_executor = CommandExecutor(self.config)
        self.system_controller = SystemController(self.config)
        self.learning_engine = LearningEngine(self.config)
        
        # État
        self.is_running = False
        self.context = {}
        self.command_queue = queue.Queue()
        
        # Initialisation
        self._initialize_components()
        
    def _setup_directories(self):
        """Créer les répertoires nécessaires"""
        self.config.BASE_DIR.mkdir(exist_ok=True)
        (self.config.BASE_DIR / "logs").mkdir(exist_ok=True)
        (self.config.BASE_DIR / "data").mkdir(exist_ok=True)
        (self.config.BASE_DIR / "scripts").mkdir(exist_ok=True)
        
    def _load_settings(self):
        """Charger les paramètres personnalisés"""
        if self.config.SETTINGS_FILE.exists():
            with open(self.config.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
                for key, value in settings.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
    def _initialize_components(self):
        """Initialiser tous les composants"""
        self.voice_engine.say("Initialisation de MEL en cours...")
        
        # Télécharger les ressources NLTK si nécessaire
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            
        # Charger le modèle spaCy
        try:
            self.nlp_processor.load_spacy_model()
        except:
            logger.warning("Modèle spaCy non trouvé")
            
        self.voice_engine.say("MEL est prêt à vous assister!")
        
    def start(self):
        """Démarrer MEL"""
        self.is_running = True
        logger.info("MEL démarré")
        
        # Thread pour l'écoute continue
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()
        
        # Thread pour l'exécution des commandes
        command_thread = threading.Thread(target=self._command_loop, daemon=True)
        command_thread.start()
        
        # Boucle principale
        try:
            while self.is_running:
                self._check_scheduled_tasks()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Arrêter MEL"""
        logger.info("Arrêt de MEL...")
        self.is_running = False
        self.voice_engine.say("Au revoir!")
        self._save_state()
        sys.exit(0)
        
    def _listen_loop(self):
        """Boucle d'écoute continue"""
        while self.is_running:
            try:
                # Écouter pour le mot de réveil
                if self.voice_engine.listen_for_wake_word():
                    self.voice_engine.play_activation_sound()
                    
                    # Écouter la commande
                    command = self.voice_engine.listen()
                    if command:
                        self.command_queue.put(command)
                        
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'écoute: {e}")
                time.sleep(1)
                
    def _command_loop(self):
        """Boucle de traitement des commandes"""
        while self.is_running:
            try:
                # Récupérer une commande de la queue
                command = self.command_queue.get(timeout=1)
                self._process_command(command)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur dans le traitement: {e}")
                self.voice_engine.say("Désolé, une erreur s'est produite")
                
    def _process_command(self, command: str):
        """Traiter une commande"""
        logger.info(f"Commande reçue: {command}")
        
        # Analyse NLP
        intent, entities = self.nlp_processor.analyze(command)
        
        # Ajout au contexte
        self.context['last_command'] = command
        self.context['last_intent'] = intent
        self.context['last_time'] = datetime.datetime.now()
        
        # Apprentissage
        self.learning_engine.record_command(command, intent, entities)
        
        # Exécution
        result = self.command_executor.execute(intent, entities, self.context)
        
        # Réponse
        if result['success']:
            if result.get('response'):
                self.voice_engine.say(result['response'])
        else:
            self.voice_engine.say(f"Désolé, je n'ai pas pu {intent}")
            
    def _check_scheduled_tasks(self):
        """Vérifier les tâches planifiées"""
        schedule.run_pending()
        
    def _save_state(self):
        """Sauvegarder l'état de MEL"""
        state = {
            'last_shutdown': datetime.datetime.now().isoformat(),
            'total_commands': self.learning_engine.get_command_count(),
            'context': self.context
        }
        
        with open(self.config.BASE_DIR / "state.json", 'w') as f:
            json.dump(state, f, indent=2)

# ==========================
# Module de Voix
# ==========================

class VoiceEngine:
    """Gestion de la reconnaissance et synthèse vocale"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Reconnaissance vocale
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.microphone = sr.Microphone()
        
        # Synthèse vocale
        self.tts_engine = pyttsx3.init()
        self._setup_voice()
        
        # Calibration
        self._calibrate_microphone()
        
    def _setup_voice(self):
        """Configurer la voix"""
        voices = self.tts_engine.getProperty('voices')
        
        # Chercher une voix française
        for voice in voices:
            if 'french' in voice.id.lower() or 'fr' in voice.id.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
                
        self.tts_engine.setProperty('rate', self.config.VOICE_RATE)
        self.tts_engine.setProperty('volume', self.config.VOICE_VOLUME)
        
    def _calibrate_microphone(self):
        """Calibrer le microphone"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
    def say(self, text: str):
        """Dire un texte"""
        logger.debug(f"MEL dit: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        
    def listen(self, timeout: int = None) -> Optional[str]:
        """Écouter et transcrire"""
        timeout = timeout or self.config.TIMEOUT_LISTENING
        
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout)
                
            # Reconnaissance
            text = self.recognizer.recognize_google(
                audio, 
                language=self.config.LANGUAGE
            )
            logger.debug(f"Transcription: {text}")
            return text.lower()
            
        except sr.UnknownValueError:
            logger.warning("Audio non compris")
            return None
        except sr.RequestError as e:
            logger.error(f"Erreur de reconnaissance: {e}")
            return None
        except sr.WaitTimeoutError:
            return None
            
    def listen_for_wake_word(self) -> bool:
        """Écouter pour le mot de réveil"""
        text = self.listen(timeout=1)
        if text and self.config.WAKE_WORD in text:
            return True
        return False
        
    def play_activation_sound(self):
        """Jouer un son d'activation"""
        # Simple beep pour l'instant
        print("\a")

# ==========================
# Module NLP
# ==========================

class NLPProcessor:
    """Traitement du langage naturel"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp = None
        
        # Intents prédéfinis
        self.intents = {
            'open_app': ['ouvre', 'lance', 'démarre', 'exécute'],
            'close_app': ['ferme', 'quitte', 'arrête', 'termine'],
            'search_web': ['cherche', 'recherche', 'google', 'trouve'],
            'system_info': ['combien', 'quel', 'état', 'info'],
            'file_operation': ['fichier', 'dossier', 'copie', 'déplace'],
            'create': ['crée', 'nouveau', 'génère', 'écris'],
            'settings': ['paramètre', 'configure', 'change', 'modifie'],
            'automation': ['automatise', 'script', 'répète', 'programme']
        }
        
    def load_spacy_model(self):
        """Charger le modèle spaCy"""
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except:
            logger.warning("Modèle spaCy non disponible")
            
    def analyze(self, text: str) -> Tuple[str, Dict]:
        """Analyser une commande"""
        # Tokenisation
        tokens = word_tokenize(text.lower())
        
        # Détection d'intention
        intent = self._detect_intent(tokens)
        
        # Extraction d'entités
        entities = self._extract_entities(text)
        
        return intent, entities
        
    def _detect_intent(self, tokens: List[str]) -> str:
        """Détecter l'intention"""
        for intent, keywords in self.intents.items():
            if any(keyword in tokens for keyword in keywords):
                return intent
                
        return 'unknown'
        
    def _extract_entities(self, text: str) -> Dict:
        """Extraire les entités"""
        entities = {
            'applications': [],
            'files': [],
            'numbers': [],
            'actions': []
        }
        
        # Extraction basique
        # Applications communes
        apps = ['chrome', 'firefox', 'word', 'excel', 'notepad', 'spotify', 'discord']
        for app in apps:
            if app in text.lower():
                entities['applications'].append(app)
                
        # Nombres
        numbers = re.findall(r'\d+', text)
        entities['numbers'] = [int(n) for n in numbers]
        
        # Si spaCy est disponible
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities['applications'].append(ent.text)
                    
        return entities

# ==========================
# Module d'Exécution
# ==========================

class CommandExecutor:
    """Exécuteur de commandes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.handlers = {
            'open_app': self._handle_open_app,
            'close_app': self._handle_close_app,
            'search_web': self._handle_search_web,
            'system_info': self._handle_system_info,
            'file_operation': self._handle_file_operation,
            'create': self._handle_create,
            'settings': self._handle_settings,
            'automation': self._handle_automation,
            'unknown': self._handle_unknown
        }
        
    def execute(self, intent: str, entities: Dict, context: Dict) -> Dict:
        """Exécuter une commande"""
        handler = self.handlers.get(intent, self._handle_unknown)
        
        try:
            return handler(entities, context)
        except Exception as e:
            logger.error(f"Erreur d'exécution: {e}")
            return {'success': False, 'error': str(e)}
            
    def _handle_open_app(self, entities: Dict, context: Dict) -> Dict:
        """Ouvrir une application"""
        apps = entities.get('applications', [])
        
        if not apps:
            return {
                'success': False,
                'response': "Quelle application voulez-vous ouvrir?"
            }
            
        app_name = apps[0]
        
        # Mapping des applications
        app_commands = {
            'chrome': 'chrome',
            'firefox': 'firefox',
            'notepad': 'notepad',
            'word': 'winword',
            'excel': 'excel',
            'spotify': 'spotify',
            'discord': 'discord'
        }
        
        command = app_commands.get(app_name)
        
        if command:
            try:
                if sys.platform == 'win32':
                    os.startfile(command)
                else:
                    subprocess.Popen([command])
                    
                return {
                    'success': True,
                    'response': f"J'ouvre {app_name}"
                }
            except:
                return {
                    'success': False,
                    'response': f"Je n'ai pas pu ouvrir {app_name}"
                }
        else:
            return {
                'success': False,
                'response': f"Je ne connais pas l'application {app_name}"
            }
            
    def _handle_close_app(self, entities: Dict, context: Dict) -> Dict:
        """Fermer une application"""
        apps = entities.get('applications', [])
        
        if not apps:
            return {
                'success': False,
                'response': "Quelle application voulez-vous fermer?"
            }
            
        app_name = apps[0]
        
        # Trouver et fermer le processus
        for proc in psutil.process_iter(['name']):
            if app_name.lower() in proc.info['name'].lower():
                try:
                    proc.terminate()
                    return {
                        'success': True,
                        'response': f"J'ai fermé {app_name}"
                    }
                except:
                    pass
                    
        return {
            'success': False,
            'response': f"Je n'ai pas trouvé {app_name}"
        }
        
    def _handle_search_web(self, entities: Dict, context: Dict) -> Dict:
        """Recherche web"""
        # Extraire la requête de recherche
        query = context.get('last_command', '').replace('cherche', '').replace('recherche', '').strip()
        
        if query:
            url = f"https://www.google.com/search?q={query}"
            webbrowser.open(url)
            
            return {
                'success': True,
                'response': f"Je recherche {query} sur Google"
            }
        else:
            return {
                'success': False,
                'response': "Que voulez-vous rechercher?"
            }
            
    def _handle_system_info(self, entities: Dict, context: Dict) -> Dict:
        """Informations système"""
        command = context.get('last_command', '')
        
        if 'batterie' in command:
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                plugged = "branchée" if battery.power_plugged else "sur batterie"
                return {
                    'success': True,
                    'response': f"La batterie est à {percent}%, {plugged}"
                }
                
        elif 'cpu' in command or 'processeur' in command:
            cpu_percent = psutil.cpu_percent(interval=1)
            return {
                'success': True,
                'response': f"Le processeur est utilisé à {cpu_percent}%"
            }
            
        elif 'mémoire' in command or 'ram' in command:
            memory = psutil.virtual_memory()
            return {
                'success': True,
                'response': f"La mémoire est utilisée à {memory.percent}%"
            }
            
        elif 'heure' in command:
            now = datetime.datetime.now()
            return {
                'success': True,
                'response': f"Il est {now.strftime('%H heures %M')}"
            }
            
        return {
            'success': False,
            'response': "Quelle information système voulez-vous?"
        }
        
    def _handle_file_operation(self, entities: Dict, context: Dict) -> Dict:
        """Opérations sur les fichiers"""
        command = context.get('last_command', '')
        
        if 'crée' in command and 'dossier' in command:
            # Extraire le nom du dossier
            words = command.split()
            if 'appelé' in words:
                idx = words.index('appelé')
                if idx + 1 < len(words):
                    folder_name = words[idx + 1]
                    path = Path.home() / 'Desktop' / folder_name
                    path.mkdir(exist_ok=True)
                    
                    return {
                        'success': True,
                        'response': f"J'ai créé le dossier {folder_name} sur le bureau"
                    }
                    
        return {
            'success': False,
            'response': "Je n'ai pas compris l'opération sur les fichiers"
        }
        
    def _handle_create(self, entities: Dict, context: Dict) -> Dict:
        """Créer du contenu"""
        command = context.get('last_command', '')
        
        if 'fichier' in command:
            # Créer un fichier texte
            if 'texte' in command:
                content = "Nouveau fichier créé par MEL\n"
                path = Path.home() / 'Desktop' / f'mel_fichier_{int(time.time())}.txt'
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                return {
                    'success': True,
                    'response': f"J'ai créé un fichier texte sur le bureau"
                }
                
        return {
            'success': False,
            'response': "Que voulez-vous créer?"
        }
        
    def _handle_settings(self, entities: Dict, context: Dict) -> Dict:
        """Gérer les paramètres"""
        command = context.get('last_command', '')
        
        if 'volume' in command:
            numbers = entities.get('numbers', [])
            if numbers:
                volume = numbers[0] / 100
                # Ajuster le volume de MEL
                return {
                    'success': True,
                    'response': f"Volume réglé à {numbers[0]}%"
                }
                
        return {
            'success': False,
            'response': "Quel paramètre voulez-vous modifier?"
        }
        
    def _handle_automation(self, entities: Dict, context: Dict) -> Dict:
        """Automatisation"""
        return {
            'success': False,
            'response': "Les fonctions d'automatisation sont en développement"
        }
        
    def _handle_unknown(self, entities: Dict, context: Dict) -> Dict:
        """Commande inconnue"""
        suggestions = [
            "Voulez-vous que j'ouvre une application?",
            "Puis-je rechercher quelque chose pour vous?",
            "Voulez-vous des informations système?"
        ]
        
        return {
            'success': False,
            'response': f"Je n'ai pas compris. {random.choice(suggestions)}"
        }

# ==========================
# Module de Contrôle Système
# ==========================

class SystemController:
    """Contrôle avancé du système"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def take_screenshot(self, path: Optional[Path] = None) -> Path:
        """Prendre une capture d'écran"""
        if not path:
            path = self.config.BASE_DIR / f"screenshot_{int(time.time())}.png"
            
        screenshot = pyautogui.screenshot()
        screenshot.save(path)
        return path
        
    def type_text(self, text: str):
        """Taper du texte"""
        pyautogui.typewrite(text, interval=0.05)
        
    def press_key(self, key: str):
        """Appuyer sur une touche"""
        pyautogui.press(key)
        
    def click_at(self, x: int, y: int):
        """Cliquer à une position"""
        pyautogui.click(x, y)
        
    def get_active_window_title(self) -> str:
        """Obtenir le titre de la fenêtre active"""
        # Implementation dépend de l'OS
        return ""
        
    def list_running_apps(self) -> List[str]:
        """Lister les applications en cours"""
        apps = []
        for proc in psutil.process_iter(['name']):
            name = proc.info['name']
            if name and name not in apps:
                apps.append(name)
        return sorted(apps)

# ==========================
# Module d'Apprentissage
# ==========================

class LearningEngine:
    """Moteur d'apprentissage et d'amélioration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.command_history = []
        self.patterns = {}
        
        self._load_history()
        
    def _load_history(self):
        """Charger l'historique"""
        if self.config.HISTORY_FILE.exists():
            with open(self.config.HISTORY_FILE, 'r') as f:
                self.command_history = json.load(f)
                
    def record_command(self, command: str, intent: str, entities: Dict):
        """Enregistrer une commande"""
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'command': command,
            'intent': intent,
            'entities': entities
        }
        
        self.command_history.append(record)
        
        # Sauvegarder périodiquement
        if len(self.command_history) % 10 == 0:
            self._save_history()
            
    def _save_history(self):
        """Sauvegarder l'historique"""
        with open(self.config.HISTORY_FILE, 'w') as f:
            json.dump(self.command_history[-1000:], f)  # Garder les 1000 dernières
            
    def get_command_count(self) -> int:
        """Nombre total de commandes"""
        return len(self.command_history)
        
    def analyze_patterns(self):
        """Analyser les patterns d'utilisation"""
        # Analyse simple des commandes fréquentes
        intents = {}
        for record in self.command_history:
            intent = record['intent']
            intents[intent] = intents.get(intent, 0) + 1
            
        return intents
        
    def suggest_automation(self) -> List[str]:
        """Suggérer des automatisations"""
        suggestions = []
        
        # Analyser les séquences répétées
        # À implémenter
        
        return suggestions

# ==========================
# Point d'Entrée Principal
# ==========================

def main():
    """Fonction principale"""
    print("""
    ╔═══════════════════════════════════════╗
    ║          MEL - Assistant IA           ║
    ║        Votre Assistant Vocal          ║
    ╚═══════════════════════════════════════╝
    """)
    
    try:
        # Créer et démarrer MEL
        mel = MEL()
        mel.start()
        
    except KeyboardInterrupt:
        print("\nArrêt de MEL...")
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}")
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
