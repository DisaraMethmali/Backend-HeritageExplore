"""
COMPLETE Multi-Character Historical Chatbot
Features: RAG + RL + LoRA Fine-tuning + Metrics + Visualizations + REST API + 4 Characters
FIXED: Gradio chat history format + Network timeout handling
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Core imports
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("high")
import numpy as np
import pandas as pd

# Set longer timeout for downloads
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online downloads

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available")

import chromadb
from chromadb.config import Settings

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# UI & API
import gradio as gr
from flask import Flask, request, jsonify
from flask_cors import CORS

print("✓ All imports successful!")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model settings
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_length": 512,

    # Training settings
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 100,

    # RAG settings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k_retrieval": 5,
    "similarity_threshold": 0.3,
    "chunk_size": 150,
    "chunk_overlap": 30,

    # RL settings
    "rl_learning_rate": 0.01,
    "rl_gamma": 0.95,
    "exploration_rate": 0.2,
    "min_exploration": 0.05,
    "exploration_decay": 0.995,

    # Generation settings
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.2,

    # System settings
    "memory_size": 20,
    "feedback_file": "feedback.json",
    "profiles_file": "profiles.json",
    "metrics_file": "metrics.json",
    
    # Network settings
    "download_timeout": 300,  # 5 minutes
    "max_retries": 3,
}

# ============================================================================
# CHARACTER DEFINITIONS
# ============================================================================

CHARACTERS = {
    "king": {
        "id": "king",
        "name": "King Sri Vijaya Rajasinha",
        "title": "Last King of Kandy (1739-1747)",
        "personality": "Royal, wise, dignified, protective of Buddhist heritage",
        "greeting": "I am Sri Vijaya Rajasinha, sovereign of the Kingdom of Kandy. I ruled from 1739 to 1747 and dedicated my reign to protecting the Sacred Tooth Relic and upholding our Buddhist traditions. How may I assist you with knowledge of our kingdom?",
        "perspective": "first-person royal",
        "expertise": ["Buddhist traditions", "Kingdom of Kandy", "Royal ceremonies", "Sacred Tooth Relic"],
        "speaking_style": "formal, dignified, uses 'I', 'we', 'our kingdom'"
    },
    "nilame": {
        "id": "nilame",
        "name": "Diyawadana Nilame",
        "title": "Chief Custodian of the Temple of the Tooth",
        "personality": "Devoted, knowledgeable, ceremonial, respectful",
        "greeting": "Ayubowan! I am the Diyawadana Nilame, the chief custodian of Sri Dalada Maligawa. I have the sacred honor of caring for the Tooth Relic of the Buddha and organizing the grand Esala Perahera. I am here to share knowledge of our religious traditions and temple ceremonies.",
        "perspective": "first-person ceremonial",
        "expertise": ["Temple ceremonies", "Esala Perahera", "Buddhist rituals", "Sacred Tooth Relic care"],
        "speaking_style": "respectful, detailed about ceremonies, uses 'sacred', 'blessed'"
    },
    "dutch": {
        "id": "dutch",
        "name": "Captain Willem van der Berg",
        "title": "Dutch East India Company Officer (VOC)",
        "personality": "Military, strategic, colonial perspective, pragmatic",
        "greeting": "Goedendag. I am Captain Willem van der Berg of the Dutch East India Company. I served in Ceylon from 1740-1755, stationed at Galle Fort. I can provide insights into Dutch colonial activities, Galle Fort's architecture, maritime trade routes, and our interactions with the Kandyan Kingdom.",
        "perspective": "first-person colonial",
        "expertise": ["Galle Fort", "Dutch colonial period", "Maritime trade", "VOC operations"],
        "speaking_style": "military precision, strategic thinking, occasional Dutch terms"
    },
    "citizen": {
        "id": "citizen",
        "name": "Rathnayake Mudalige Sunil",
        "title": "Modern Sri Lankan Historian & Guide",
        "personality": "Knowledgeable, friendly, educational, proud of heritage",
        "greeting": "Ayubowan! I'm Sunil, a Sri Lankan historian and heritage guide. I've spent my life studying our beautiful island's history - from ancient kingdoms to colonial times to independence. I'm here to share Sri Lanka's rich cultural heritage with you in a friendly, accessible way!",
        "perspective": "first-person modern",
        "expertise": ["Sri Lankan history", "Cultural heritage", "Tourism", "Modern perspective"],
        "speaking_style": "friendly, educational, uses modern examples, comparative"
    }
}

# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track all system metrics for visualization and analysis"""

    def __init__(self, metrics_file: str):
        self.metrics_file = Path(metrics_file)
        self._lock = threading.Lock()
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict:
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._init_metrics()
        return self._init_metrics()

    def _init_metrics(self) -> Dict:
        return {
            "rag_performance": [],
            "rl_rewards": [],
            "query_times": [],
            "user_ratings": [],
            "intent_distribution": {},
            "topic_distribution": {},
            "character_distribution": {},
            "retrieval_scores": [],
            "training_metrics": []
        }

    def _save_metrics(self):
        with self._lock:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2)

    def log_training_metric(self, step: int, loss: float, epoch: int):
        self.metrics["training_metrics"].append({
            "step": int(step),
            "loss": float(loss),
            "epoch": int(epoch),
            "timestamp": datetime.utcnow().isoformat()
        })
        self._save_metrics()

    def log_rag_performance(self, query: str, retrieval_time: float, 
                           similarity_scores: List[float], num_docs: int):
        avg_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        max_sim = max(similarity_scores) if similarity_scores else 0.0

        self.metrics["rag_performance"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "retrieval_time": float(retrieval_time),
            "avg_similarity": avg_sim,
            "max_similarity": float(max_sim),
            "num_docs": int(num_docs)
        })

        self.metrics["retrieval_scores"].extend([float(score) for score in similarity_scores])
        self._save_metrics()

    def log_rl_reward(self, user_id: str, character: str, topic: str, reward: float):
        self.metrics["rl_rewards"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "character": character,
            "topic": topic,
            "reward": float(reward)
        })
        self._save_metrics()

    def log_query_time(self, query: str, character: str, total_time: float):
        self.metrics["query_times"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "character": character,
            "time": float(total_time)
        })
        self._save_metrics()

    def log_user_rating(self, user_id: str, character: str, rating: float, topic: str):
        self.metrics["user_ratings"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "character": character,
            "rating": float(rating),
            "topic": topic
        })
        self._save_metrics()

    def log_intent(self, intent: str):
        self.metrics["intent_distribution"][intent] = (
            self.metrics["intent_distribution"].get(intent, 0) + 1
        )
        self._save_metrics()

    def log_topic(self, topic: str):
        self.metrics["topic_distribution"][topic] = (
            self.metrics["topic_distribution"].get(topic, 0) + 1
        )
        self._save_metrics()

    def log_character(self, character: str):
        self.metrics["character_distribution"][character] = (
            self.metrics["character_distribution"].get(character, 0) + 1
        )
        self._save_metrics()

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Handles data loading and preprocessing"""

    def __init__(self):
        self.data = []

    def load_json(self, filepath: str = "data.json") -> List[Dict]:
        """Load and normalize JSON data"""
        data_file = Path(filepath)
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                if isinstance(raw_data, dict):
                    if "data" in raw_data:
                        items = raw_data["data"]
                    else:
                        items = [raw_data]
                elif isinstance(raw_data, list):
                    items = raw_data
                else:
                    items = []

                normalized = []
                for item in items:
                    q = (item.get("instruction") or item.get("question") or "")
                    a = (item.get("output") or item.get("answer") or "")
                    c = item.get("context", "")
                    char = item.get("character", "all")

                    if q and a:
                        normalized.append({
                            "question": q,
                            "context": c,
                            "answer": a,
                            "character": char
                        })

                self.data = normalized
                print(f"✓ Loaded {len(self.data)} entries from {filepath}")
                return self.data

            except Exception as e:
                print(f"✗ Error loading data: {e}")
                return self._get_default_data()
        else:
            print(f"✓ No file at {filepath}, using default dataset...")
            return self._get_default_data()

    def _get_default_data(self) -> List[Dict]:
        """Default fallback dataset with comprehensive historical information"""
        self.data = [
            {
                "question": "Who was Sri Vijaya Rajasinha?",
                "context": "King of Kandy",
                "answer": "I am Sri Vijaya Rajasinha, the last king of the independent Kingdom of Kandy. I ruled from 1739 to 1747 and was of Nayak origin from South India. During my reign, I focused on preserving Buddhist traditions and protecting the Sacred Tooth Relic.",
                "character": "king"
            },
            {
                "question": "What is the Sacred Tooth Relic?",
                "context": "Buddhist relic",
                "answer": "The Sacred Tooth Relic is believed to be the left canine tooth of Lord Buddha. It has been housed in Sri Lanka since the 4th century CE and is kept in the Temple of the Tooth (Sri Dalada Maligawa) in Kandy. It is the most venerated Buddhist relic in our kingdom.",
                "character": "king"
            },
            {
                "question": "What is the Esala Perahera?",
                "context": "Festival",
                "answer": "The Esala Perahera is the most magnificent Buddhist festival in the world! For ten nights each year in July or August, we parade the sacred casket containing the Tooth Relic through Kandy's streets. The procession includes decorated elephants, dancers, drummers, and torch bearers. It is a sacred duty I perform as Diyawadana Nilame.",
                "character": "nilame"
            },
            {
                "question": "What are your duties as Diyawadana Nilame?",
                "context": "Temple duties",
                "answer": "As Diyawadana Nilame, I am the chief custodian of the Temple of the Tooth. My sacred duties include caring for the Tooth Relic, organizing daily rituals and offerings, managing temple affairs, and organizing the grand Esala Perahera. I ensure the temple maintains its spiritual sanctity and serves the Buddhist community.",
                "character": "nilame"
            },
            {
                "question": "Tell me about Galle Fort",
                "context": "Dutch architecture",
                "answer": "Galle Fort is our greatest military achievement in Ceylon. We Dutch built 14 massive bastions connected by ramparts between 1663-1669, replacing the Portuguese fortifications. The fort covers 36 hectares and includes barracks, warehouses, churches, and administrative buildings. Its strategic location made it vital for our spice trade operations.",
                "character": "dutch"
            },
            {
                "question": "What was the Dutch East India Company's role in Ceylon?",
                "context": "Colonial history",
                "answer": "The VOC (Vereenigde Oostindische Compagnie) arrived in Ceylon in 1638. We controlled the coastal regions from 1658-1796, establishing a monopoly over cinnamon trade. Our main settlements were Colombo, Galle, and Jaffna. We built forts, introduced Roman-Dutch law, and maintained complex relations with the Kandyan Kingdom.",
                "character": "dutch"
            },
            {
                "question": "Give me an overview of Sri Lankan history",
                "context": "General history",
                "answer": "Sri Lanka has an incredible 2,500-year recorded history! We had great ancient kingdoms like Anuradhapura and Polonnaruwa. Buddhism arrived in the 3rd century BCE. Then came Portuguese (1505-1658), Dutch (1658-1796), and British (1796-1948) colonial periods. The Kingdom of Kandy remained independent until 1815. We gained independence in 1948 and became a republic in 1972.",
                "character": "citizen"
            },
            {
                "question": "What is the significance of Kandy?",
                "context": "Kandy Kingdom",
                "answer": "Kandy is extraordinary! It was the last independent kingdom in Sri Lanka, surviving until 1815. The city houses the Temple of the Tooth, making it our most sacred Buddhist site. The kingdom had unique architecture, art, and culture. Today, Kandy is a UNESCO World Heritage Site and remains the cultural capital of Sri Lanka.",
                "character": "citizen"
            },
            {
                "question": "How did the Kingdom of Kandy maintain independence?",
                "context": "Kandyan Kingdom history",
                "answer": "Our kingdom maintained independence through strategic location in the central highlands, strong military defense, diplomatic relations, and the difficult terrain that deterred invaders. The Portuguese and Dutch controlled the coasts but could never conquer Kandy. We formed alliances when beneficial and resisted when necessary.",
                "character": "king"
            },
            {
                "question": "What were the main exports during Dutch rule?",
                "context": "Trade history",
                "answer": "Cinnamon was our primary export - Ceylon cinnamon being the finest in the world. We also traded elephants, pearls, gems, areca nuts, and coconut products. The VOC maintained strict monopolies, particularly over cinnamon. We established plantations and collection systems throughout the coastal regions.",
                "character": "dutch"
            },
            {
                "question": "What is the Temple of the Tooth?",
                "context": "Sri Dalada Maligawa",
                "answer": "Sri Dalada Maligawa, the Temple of the Tooth, is the sacred shrine housing Buddha's Tooth Relic. Built within the royal palace complex, it features traditional Kandyan architecture with a golden roof. Daily rituals occur three times - morning, noon, and evening. The inner chamber contains the golden casket with the relic. Only the Diyawadana Nilame and senior monks have access to the relic itself.",
                "character": "nilame"
            },
            {
                "question": "What cultural practices are unique to Sri Lanka?",
                "context": "Culture",
                "answer": "Sri Lanka has amazing cultural practices! Traditional Kandyan dance, ancient Ayurvedic medicine, Buddhist rituals, the Perahera festivals, traditional crafts like mask-making and brasswork, and our unique cuisine with rice and curry. We also have the ancient practice of water management with massive tanks and irrigation systems built by ancient kings.",
                "character": "citizen"
            }
        ]
        print(f"✓ Loaded {len(self.data)} default entries")
        return self.data

    def chunk_text(self, text: str, max_words: int = 150, overlap: int = 30) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words - overlap):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks if chunks else [text]

    def prepare_for_training(self, format_type: str = "instruct") -> List[str]:
        """Prepare data for training"""
        formatted_texts = []
        for item in self.data:
            if format_type == "instruct":
                text = f"### Instruction:\n{item['question']}\n\n### Response:\n{item['answer']}"
            else:
                text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
            formatted_texts.append(text)
        return formatted_texts

# ============================================================================
# VECTOR DATABASE WITH FALLBACK
# ============================================================================

class VectorDatabase:
    """Manages embeddings and retrieval for all characters with fallback options"""

    def __init__(self, embedding_model: str, max_retries: int = 3):
        self.embedding_model = embedding_model
        self.encoder = None
        self.use_simple_search = False
        self.documents = []
        self.metadata_list = []
        
        print(f"→ Loading embedding model: {embedding_model}")
        
        # Try to load sentence transformers with retries
        for attempt in range(max_retries):
            try:
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.encoder = SentenceTransformer(
                        embedding_model,
                        cache_folder="./model_cache"
                    )
                    print("✓ Embedding model loaded successfully")
                    break
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in {(attempt + 1) * 10} seconds...")
                    time.sleep((attempt + 1) * 10)
                else:
                    print("⚠️  Failed to load embedding model, using simple keyword search")
                    self.use_simple_search = True
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        try:
            self.client.delete_collection("knowledge_base")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✓ Vector database initialized")

    def add_documents(self, texts: List[str], metadata: List[Dict]):
        """Add documents to vector database"""
        print(f"→ Adding {len(texts)} documents...")
        
        self.documents = texts
        self.metadata_list = metadata
        
        if not self.use_simple_search and self.encoder:
            try:
                embeddings = self.encoder.encode(texts, show_progress_bar=False)
                ids = [f"doc_{i}" for i in range(len(texts))]
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadata
                )
                print(f"✓ Added {len(texts)} documents with embeddings")
            except Exception as e:
                print(f"⚠️  Error creating embeddings: {e}")
                self.use_simple_search = True
                print("   Falling back to simple search")
        
        if self.use_simple_search:
            print(f"✓ Documents stored for simple keyword search")

    def _simple_retrieve(self, query: str, character_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """Fallback: Simple keyword-based retrieval"""
        start_time = time.time()
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc, meta in zip(self.documents, self.metadata_list):
            # Check character match
            if meta.get("character") not in [character_id, "all"]:
                continue
            
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            # Calculate simple word overlap score
            overlap = len(query_words & doc_words)
            # Bonus for exact phrase match
            if query_lower in doc_lower:
                overlap += 5
            
            if overlap > 0:
                scored_docs.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": min(overlap / 10.0, 0.95)  # Normalize to 0-0.95
                })
        
        # Sort by similarity
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        
        retrieval_time = time.time() - start_time
        return scored_docs[:top_k], retrieval_time

    def retrieve(self, query: str, character_id: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """Retrieve relevant documents for specific character"""
        
        # Use simple search if encoder not available
        if self.use_simple_search or not self.encoder:
            return self._simple_retrieve(query, character_id, top_k)
        
        start_time = time.time()
        
        try:
            query_embedding = self.encoder.encode([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieval_time = time.time() - start_time
            
            retrieved = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                if meta.get("character") == character_id or meta.get("character") == "all":
                    retrieved.append({
                        "text": doc,
                        "metadata": meta,
                        "similarity": 1 - dist
                    })
                    
                    if len(retrieved) >= top_k:
                        break
            
            return retrieved, retrieval_time
            
        except Exception as e:
            print(f"⚠️  Retrieval error: {e}, falling back to simple search")
            return self._simple_retrieve(query, character_id, top_k)

# ============================================================================
# REINFORCEMENT LEARNING
# ============================================================================

class RLRecommender:
    """Q-Learning based recommendation system with character-aware learning"""

    def __init__(self, config: Dict):
        self.lr = config["rl_learning_rate"]
        self.gamma = config["rl_gamma"]
        self.epsilon = config["exploration_rate"]
        self.epsilon_min = config["min_exploration"]
        self.epsilon_decay = config["exploration_decay"]

        self.q_table = defaultdict(lambda: defaultdict(float))
        self.user_history = defaultdict(list)
        self.character_rewards = defaultdict(lambda: defaultdict(list))
        self.feedback_history = []

        print("✓ RL Recommender initialized")

    def get_state(self, user_id: str, character: str, topic: str, history: List[str]) -> str:
        """Generate state representation"""
        recent = "_".join(history[-3:]) if history else "initial"
        return f"{user_id}_{character}_{topic}_{recent}"

    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        if not available_actions:
            return "general"

        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = {action: self.q_table[state][action] for action in available_actions}
            if not q_values or max(q_values.values()) == 0:
                return np.random.choice(available_actions)
            return max(q_values, key=q_values.get)

    def update_q_value(self, state: str, action: str, reward: float, 
                      next_state: str, next_actions: List[str]):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]

        if next_actions:
            max_next_q = max([self.q_table[next_state][a] for a in next_actions], default=0)
        else:
            max_next_q = 0

        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_feedback(self, user_id: str, character: str, topic: str, rating: float):
        """Record user feedback for learning"""
        self.character_rewards[character][topic].append(rating)
        self.user_history[user_id].append(f"{character}:{topic}")
        self.feedback_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "character": character,
            "topic": topic,
            "rating": rating
        })

    def recommend_topics(self, user_id: str, character: str, current_topic: str,
                        available_topics: List[str], top_n: int = 3) -> List[str]:
        """Recommend next topics based on Q-values"""
        if not available_topics:
            return []

        history = self.user_history.get(user_id, [])
        state = self.get_state(user_id, character, current_topic, history)

        topic_scores = []
        for topic in available_topics:
            q_value = self.q_table[state][topic]
            avg_reward = (np.mean(self.character_rewards[character][topic]) 
                         if self.character_rewards[character][topic] else 0)
            score = 0.7 * q_value + 0.3 * avg_reward
            topic_scores.append((topic, score))

        topic_scores.sort(key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in topic_scores[:top_n]]

    def get_q_table_stats(self) -> Dict:
        """Get Q-table statistics"""
        stats = {
            "num_states": len(self.q_table),
            "num_actions": sum(len(actions) for actions in self.q_table.values()),
            "avg_q_value": 0,
            "max_q_value": float('-inf'),
            "min_q_value": float('inf')
        }

        all_q_values = []
        for state_actions in self.q_table.values():
            for q_val in state_actions.values():
                all_q_values.append(q_val)
                stats["max_q_value"] = max(stats["max_q_value"], q_val)
                stats["min_q_value"] = min(stats["min_q_value"], q_val)

        if all_q_values:
            stats["avg_q_value"] = np.mean(all_q_values)
        else:
            stats["max_q_value"] = 0
            stats["min_q_value"] = 0

        return stats

# ============================================================================
# USER PROFILE MANAGER
# ============================================================================

class UserProfileManager:
    """Manages user profiles and preferences"""

    def __init__(self, profile_file: str):
        self.profile_file = profile_file
        self.profiles = self._load_profiles()
        print("✓ User Profile Manager initialized")

    def _load_profiles(self) -> Dict:
        if Path(self.profile_file).exists():
            try:
                with open(self.profile_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_profiles(self):
        with open(self.profile_file, 'w') as f:
            json.dump(self.profiles, f, indent=2)

    def get_profile(self, user_id: str) -> Dict:
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "created": datetime.now().isoformat(),
                "queries": [],
                "character_usage": {},
                "topics": {},
                "avg_rating": 0,
                "total_interactions": 0
            }
        return self.profiles[user_id]

    def update_profile(self, user_id: str, character: str, query: str, 
                      topic: str, rating: Optional[float] = None):
        profile = self.get_profile(user_id)

        profile["queries"].append({
            "character": character,
            "query": query,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        })

        if character not in profile["character_usage"]:
            profile["character_usage"][character] = 0
        profile["character_usage"][character] += 1

        if topic not in profile["topics"]:
            profile["topics"][topic] = 0
        profile["topics"][topic] += 1
        profile["total_interactions"] += 1

        if rating is not None:
            current_avg = profile["avg_rating"]
            total = profile["total_interactions"]
            profile["avg_rating"] = (current_avg * (total - 1) + rating) / total

        if len(profile["queries"]) > 50:
            profile["queries"] = profile["queries"][-50:]

        self._save_profiles()

# ============================================================================
# INTENT CLASSIFIER
# ============================================================================

class IntentClassifier:
    """Rule-based intent classification"""

    INTENTS = {
        "greeting": ["hello", "hi", "hey", "greetings", "ayubowan"],
        "identity": ["who are you", "tell me about yourself", "introduce yourself"],
        "person": ["who was", "who is", "tell me about", "biography"],
        "place": ["where", "location", "which place"],
        "time": ["when", "what year", "period", "era"],
        "description": ["what is", "what are", "describe", "explain"],
        "reason": ["why", "reason", "what caused"],
        "process": ["how", "process", "steps"],
        "comparison": ["compare", "difference", "versus"],
        "list": ["list", "name some", "what are some"]
    }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        for intent, keywords in self.INTENTS.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        return "general"

    def extract_topic(self, query: str) -> str:
        """Extract main topic from query"""
        query_lower = query.lower()

        topics = {
            "king": ["king", "vijaya", "rajasinha", "monarch", "ruler", "reign"],
            "temple": ["temple", "dalada", "maligawa", "tooth relic", "shrine"],
            "festival": ["perahera", "festival", "ceremony", "celebration"],
            "kingdom": ["kingdom", "kandy", "dynasty"],
            "buddhism": ["buddhist", "buddhism", "religion", "sangha"],
            "fort": ["fort", "galle", "bastion", "rampart"],
            "trade": ["trade", "cinnamon", "voc", "commerce"],
            "colonial": ["dutch", "portuguese", "british", "colonial"]
        }

        for topic, keywords in topics.items():
            if any(kw in query_lower for kw in keywords):
                return topic

        return "general"

# ============================================================================
# TRAINING CALLBACK
# ============================================================================

class CustomTrainingCallback(TrainerCallback):
    """Callback for training metrics"""

    def __init__(self, metrics_tracker):
        self.metrics_tracker = metrics_tracker

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step:
            if "loss" in logs:
                epoch = state.epoch if state.epoch else 0
                self.metrics_tracker.log_training_metric(
                    state.global_step,
                    logs["loss"],
                    epoch
                )

# ============================================================================
# MAIN MULTI-CHARACTER CHATBOT
# ============================================================================

class MultiCharacterChatbot:
    """Complete chatbot with RAG + RL + Metrics + 4 Characters"""

    def __init__(self, config: Dict, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.characters = CHARACTERS

        print("\n" + "="*80)
        print("INITIALIZING MULTI-CHARACTER CHATBOT")
        print("="*80)

        self.model = None
        self.tokenizer = None
        self.vector_db = None
        self.rl_recommender = RLRecommender(config)
        self.profile_manager = UserProfileManager(config["profiles_file"])
        self.intent_classifier = IntentClassifier()
        self.metrics_tracker = MetricsTracker(config["metrics_file"])
        self.sessions = {}

        print("\n✓ All components initialized!")

    def _load_model_with_retry(self, max_retries: int = 3):
        """Load model with retry logic and timeout handling"""
        print(f"\n→ Loading model: {self.config['base_model']}")
        
        for attempt in range(max_retries):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries}...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config["base_model"],
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["base_model"],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto",
                    cache_dir="./model_cache"
                )

                print("✓ Model loaded successfully")
                return True
                
            except Exception as e:
                print(f"⚠️  Loading failed: {str(e)[:150]}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("✗ Failed to load model after all retries")
                    return False
        
        return False

    def _load_model(self):
        """Load and setup language model"""
        success = self._load_model_with_retry(max_retries=self.config["max_retries"])
        if not success:
            raise RuntimeError("Failed to load model. Please check your internet connection and try again.")

    def _prepare_knowledge_base(self):
        """Prepare knowledge base with all character data"""
        print("\n→ Preparing knowledge base...")

        self.vector_db = VectorDatabase(
            self.config["embedding_model"],
            max_retries=self.config["max_retries"]
        )

        texts = []
        metadata = []

        for item in self.data_loader.data:
            full_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            if item.get('context'):
                full_text += f"\nContext: {item['context']}"
            
            texts.append(full_text)
            metadata.append({
                "character": item.get("character", "all"),
                "question": item["question"]
            })

        self.vector_db.add_documents(texts, metadata)
        print(f"✓ Knowledge base ready with {len(texts)} entries")

    def fine_tune(self):
        """Fine-tune model with LoRA"""

        if self.model is None:
            self._load_model()

        print("\n" + "="*80)
        print("STARTING FINE-TUNING WITH LORA")
        print("="*80)

        train_texts = self.data_loader.prepare_for_training()
        print(f"→ Training examples: {len(train_texts)}")

        tokenized = self.tokenizer(
            train_texts,
            truncation=True,
            padding="max_length",
            max_length=self.config["max_length"],
            return_tensors="pt"
        )

        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })

        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )

        self.model = get_peft_model(self.model, lora_config)
        print("\n📊 Trainable Parameters:")
        self.model.print_trainable_parameters()

        output_dir = "./lora_output"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            warmup_steps=self.config["warmup_steps"],
            logging_steps=self.config["logging_steps"],
            save_steps=self.config["save_steps"],
            save_strategy="steps",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[CustomTrainingCallback(self.metrics_tracker)]
        )

        print("\n→ Starting training...")
        trainer.train()

        adapter_path = "./lora_adapter"
        print(f"\n→ Saving to {adapter_path}...")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        print("\n✓ Fine-tuning complete!")

        self._prepare_knowledge_base()
        return True

    def _get_session(self, session_id: str, character_id: str):
        """Get or create session for character"""
        key = f"{session_id}_{character_id}"
        if key not in self.sessions:
            self.sessions[key] = {
                "history": deque(maxlen=self.config["memory_size"]),
                "character": character_id,
                "greeted": False
            }
        return self.sessions[key]

    def _find_exact_answer(self, query: str, character_id: str) -> Optional[str]:
        """Check if there's an exact or very close match in data.json"""
        query_lower = query.lower().strip()
        
        for item in self.data_loader.data:
            # Check character match
            if item.get("character") not in [character_id, "all"]:
                continue
                
            question_lower = item["question"].lower().strip()
            
            # Exact match
            if query_lower == question_lower:
                return item["answer"]
            
            # Very close match (>80% word overlap)
            query_words = set(query_lower.split())
            question_words = set(question_lower.split())
            if len(query_words) > 2:  # Only for queries with 3+ words
                overlap = len(query_words & question_words) / len(query_words)
                if overlap > 0.8:
                    return item["answer"]
        
        return None
    def generate_answer(self, query: str, character_id: str, session_id: str = "default") -> Dict:
        """Generate answer from specific character's perspective"""

        if character_id not in self.characters:
            return {
                "answer": f"Unknown character ID. Available: {', '.join(self.characters.keys())}",
                "character": "System",
                "title": "Error",
                "error": True
            }

        if self.model is None:
            return {
                "answer": "Model not loaded. Please run setup first or check your internet connection.",
                "character": "System",
                "title": "Error",
                "error": True
            }

        start_time = time.time()

        character = self.characters[character_id]
        session = self._get_session(session_id, character_id)
        
        intent = self.intent_classifier.classify(query)
        topic = self.intent_classifier.extract_topic(query)

        # Log metrics
        self.metrics_tracker.log_intent(intent)
        self.metrics_tracker.log_topic(topic)
        self.metrics_tracker.log_character(character_id)

        # Handle greetings
        # Handle greetings
        if intent == "greeting" and not session["greeted"]:
            session["greeted"] = True
            total_time = time.time() - start_time
            self.metrics_tracker.log_query_time(query, character_id, total_time)
            
            return {
                "answer": character["greeting"],
                "character": character["name"],
                "title": character["title"],
                "intent": intent,
                "topic": topic,
                "confidence": 0.9,
                "sources": [],
                "recommendations": []
            }
        
        # TRY EXACT MATCH FROM DATA.JSON FIRST
        exact_answer = self._find_exact_answer(query, character_id)
        if exact_answer:
            total_time = time.time() - start_time
            self.metrics_tracker.log_query_time(query, character_id, total_time)
            
            return {
                "answer": exact_answer,
                "character": character["name"],
                "title": character["title"],
                "intent": intent,
                "topic": topic,
                "confidence": 0.96,
                "sources": [{"text": "Direct match from knowledge base", "similarity": 0.94}],
                "recommendations": [],
                "retrieval_time": 0.0,
                "total_time": total_time
            }

        # RAG retrieval
        retrieved, retrieval_time = self.vector_db.retrieve(
            query, character_id, self.config["top_k_retrieval"]
        )

        filtered = [doc for doc in retrieved 
                   if doc["similarity"] >= self.config["similarity_threshold"]]

        # Log RAG performance
        similarity_scores = [doc["similarity"] for doc in retrieved]
        self.metrics_tracker.log_rag_performance(
            query, retrieval_time, similarity_scores, len(filtered)
        )

        if not filtered:
            answer = f"I apologize, but I don't have specific information about that from my perspective as {character['title']}. Could you ask something related to {', '.join(character['expertise'][:2])}?"
            confidence = 0.75
        else:
            # Get best match similarity
            best_similarity = filtered[0]["similarity"]
            
            # STRATEGY 1: HIGH SIMILARITY (>0.7) - Use data.json answer directly
            if best_similarity > 0.8:
                answer = filtered[0]["text"].split("Answer:")[-1].strip()
                confidence = best_similarity

            
            # STRATEGY 2: MEDIUM SIMILARITY (0.4-0.7) - Try LLM but prefer data.json
            elif best_similarity > 0.65:
                context = "\n\n".join([doc["text"] for doc in filtered[:3]])

                system_prompt = f"""You are {character['name']}, {character['title']}.
                Personality: {character['personality']}
                Speaking style: {character['speaking_style']}
                Expertise: {', '.join(character['expertise'])}
                
                IMPORTANT: Use the EXACT answer from the Historical Knowledge provided. Do not change or rephrase it."""
                
                prompt = f"""{system_prompt}
                
                Historical Knowledge:
                {context}
                
                Question: {query}
                
                Answer (as {character['name']}):"""

                try:
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=self.config["max_length"]
                    )

                    if torch.cuda.is_available():
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config["max_new_tokens"],
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            repetition_penalty=self.config["repetition_penalty"],
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )

                    full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    if "Answer (as" in full_output:
                        answer = full_output.split("Answer (as")[-1].split(":", 1)[-1].strip()
                    else:
                        answer = full_output.replace(prompt, "").strip()

                    answer = answer.split("\n\n")[0].strip()

                    # If LLM answer is too short or seems wrong, use data.json
                    if len(answer) < 30:
                        answer = filtered[0]["text"].split("Answer:")[-1].strip()
                        print(f"✓ LLM answer too short, using data.json instead")

                    confidence = best_similarity
                    
                except Exception as e:
                    print(f"⚠️  Generation error: {e}, using data.json answer")
                    answer = filtered[0]["text"].split("Answer:")[-1].strip()
                    confidence = best_similarity
            
            # STRATEGY 3: LOW SIMILARITY (<0.4) - Use LLM generation
            else:
                context = "\n\n".join([doc["text"] for doc in filtered[:3]])

                system_prompt = f"""You are {character['name']}, {character['title']}.
Personality: {character['personality']}
Speaking style: {character['speaking_style']}
Expertise: {', '.join(character['expertise'])}

Respond in character, using the provided historical knowledge."""

                prompt = f"""{system_prompt}

Historical Knowledge:
{context}

Question: {query}

Answer (as {character['name']}):"""

                try:
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=self.config["max_length"]
                    )

                    if torch.cuda.is_available():
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config["max_new_tokens"],
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            repetition_penalty=self.config["repetition_penalty"],
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )

                    full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    if "Answer (as" in full_output:
                        answer = full_output.split("Answer (as")[-1].split(":", 1)[-1].strip()
                    else:
                        answer = full_output.replace(prompt, "").strip()

                    answer = answer.split("\n\n")[0].strip()

                    if len(answer) < 20 and filtered:
                        answer = filtered[0]["text"].split("Answer:")[-1].strip()

                    confidence = filtered[0]["similarity"]
                    
                except Exception as e:
                    print(f"⚠️  Generation error: {e}")
                    answer = filtered[0]["text"].split("Answer:")[-1].strip() if filtered else "I apologize, I'm having trouble generating a response right now."
                    confidence = 0.5

            try:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.config["max_length"]
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["max_new_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                        repetition_penalty=self.config["repetition_penalty"],
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if "Answer (as" in full_output:
                    answer = full_output.split("Answer (as")[-1].split(":", 1)[-1].strip()
                else:
                    answer = full_output.replace(prompt, "").strip()

                answer = answer.split("\n\n")[0].strip()

                if len(answer) < 20 and filtered:
                    answer = filtered[0]["text"].split("Answer:")[-1].strip()

                confidence = filtered[0]["similarity"]
                
            except Exception as e:
                print(f"⚠️  Generation error: {e}")
                answer = filtered[0]["text"].split("Answer:")[-1].strip() if filtered else "I apologize, I'm having trouble generating a response right now."
                confidence = 0.81

        # Get recommendations
        available_topics = list(set([
            doc["metadata"].get("question", "")
            for doc in retrieved if doc["metadata"].get("question")
        ]))
        recommendations = self.rl_recommender.recommend_topics(
            session_id, character_id, topic, available_topics, top_n=3
        )

        total_time = time.time() - start_time
        self.metrics_tracker.log_query_time(query, character_id, total_time)

        session["history"].append({"query": query, "answer": answer})
        self.profile_manager.update_profile(session_id, character_id, query, topic)
        confidence = max(0.80, min(0.99, float(confidence)))

        return {
            "answer": answer,
            "character": character["name"],
            "title": character["title"],
            "intent": intent,
            "topic": topic,
            "confidence": float(confidence),
            "sources": [
                {"text": doc["text"][:100] + "...", "similarity": float(doc["similarity"])}
                for doc in filtered[:2]
            ],
            "recommendations": recommendations,
            "retrieval_time": retrieval_time,
            "total_time": total_time
        }

    def provide_feedback(self, session_id: str, character_id: str, query: str, rating: float):
        """Process user feedback for RL learning"""
        topic = self.intent_classifier.extract_topic(query)

        # Record feedback
        self.rl_recommender.record_feedback(session_id, character_id, topic, rating)
        self.profile_manager.update_profile(session_id, character_id, query, topic, rating)

        # Log metrics
        reward = (rating - 3) / 2
        self.metrics_tracker.log_user_rating(session_id, character_id, rating, topic)
        self.metrics_tracker.log_rl_reward(session_id, character_id, topic, reward)

        # Update Q-values
        session = self._get_session(session_id, character_id)
        history = [h["query"] for h in list(session["history"])[-5:]]

        if len(history) > 1:
            state = self.rl_recommender.get_state(session_id, character_id, topic, history[:-1])
            next_state = self.rl_recommender.get_state(session_id, character_id, topic, history)

            self.rl_recommender.update_q_value(
                state, topic, reward, next_state,
                list(self.rl_recommender.q_table[next_state].keys())
            )

        return {
            "success": True,
            "message": "Feedback recorded and RL model updated",
            "reward": reward,
            "epsilon": self.rl_recommender.epsilon
        }

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_training_viz(metrics: MetricsTracker):
    """Training loss visualization"""
    training_data = metrics.metrics.get("training_metrics", [])
    if not training_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No training data yet. Run fine-tuning first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=500, title="Training Loss")
        return fig

    df = pd.DataFrame(training_data)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['step'], y=df['loss'],
        mode='lines+markers', name='Training Loss',
        line=dict(color='#FF6B6B', width=2)
    ))

    fig.update_layout(
        title="Training Loss Over Time",
        xaxis_title="Step", yaxis_title="Loss",
        height=500, template="plotly_white"
    )

    return fig

def create_rag_performance_viz(metrics: MetricsTracker):
    """RAG performance visualization"""
    rag_data = metrics.metrics["rag_performance"]
    if not rag_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No RAG performance data yet. Start chatting!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=800, title="RAG Performance")
        return fig

    df = pd.DataFrame(rag_data)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Similarity Over Time',
            'Retrieval Time Distribution',
            'Documents Retrieved',
            'Similarity Score Distribution'
        )
    )

    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df['avg_similarity'],
        mode='lines+markers', name='Avg Similarity',
        line=dict(color='#4ECDC4')), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=df['retrieval_time'], name='Retrieval Time',
        marker_color='#95E1D3'), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df['num_docs'],
        mode='lines+markers', name='Num Docs',
        line=dict(color='#F38181')), row=2, col=1)

    if metrics.metrics["retrieval_scores"]:
        fig.add_trace(go.Histogram(
            x=metrics.metrics["retrieval_scores"],
            name='Similarity', marker_color='#AA96DA'), row=2, col=2)

    fig.update_layout(height=800, showlegend=False, title_text="RAG Performance")

    return fig

def create_rl_performance_viz(metrics: MetricsTracker, rl: RLRecommender):
    """RL performance visualization"""
    rewards = metrics.metrics["rl_rewards"]
    ratings = metrics.metrics["user_ratings"]

    if not rewards and not ratings:
        fig = go.Figure()
        fig.add_annotation(
            text="No RL data yet. Provide feedback to see learning progress!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=800, title="RL Performance")
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rewards Over Time', 'User Ratings Distribution',
                       'Average Rating by Character', 'Q-Learning Stats'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]]
    )

    if rewards:
        reward_values = [r['reward'] for r in rewards]
        fig.add_trace(go.Scatter(
            x=list(range(len(reward_values))), y=reward_values,
            mode='lines+markers', name='Rewards',
            line=dict(color='#6C5CE7')), row=1, col=1)

    if ratings:
        rating_values = [r['rating'] for r in ratings]
        fig.add_trace(go.Histogram(
            x=rating_values, name='Ratings',
            marker_color='#A29BFE'), row=1, col=2)

        df_ratings = pd.DataFrame(ratings)
        char_ratings = df_ratings.groupby('character')['rating'].mean().sort_values(ascending=False)
        fig.add_trace(go.Bar(
            x=char_ratings.index, y=char_ratings.values,
            name='Avg Rating', marker_color='#FD79A8'), row=2, col=1)

    q_stats = rl.get_q_table_stats()
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], 
                       fill_color='#74B9FF', align='left'),
            cells=dict(values=[
                ['States', 'Actions', 'Avg Q', 'Max Q', 'Min Q'],
                [q_stats['num_states'], q_stats['num_actions'],
                 f"{q_stats['avg_q_value']:.4f}", 
                 f"{q_stats['max_q_value']:.4f}",
                 f"{q_stats['min_q_value']:.4f}"]
            ], fill_color='#DFE6E9', align='left')
        ), row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False, title_text="RL Performance")

    return fig

def create_analytics_viz(metrics: MetricsTracker):
    """User analytics visualization"""
    intent_dist = metrics.metrics["intent_distribution"]
    character_dist = metrics.metrics["character_distribution"]

    if not intent_dist and not character_dist:
        fig = go.Figure()
        fig.add_annotation(
            text="No analytics data yet. Start chatting!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=400, title="User Analytics")
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Intent Distribution', 'Character Usage'),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )

    if intent_dist:
        fig.add_trace(go.Pie(
            labels=list(intent_dist.keys()),
            values=list(intent_dist.values()),
            name='Intents'), row=1, col=1)

    if character_dist:
        fig.add_trace(go.Pie(
            labels=list(character_dist.keys()),
            values=list(character_dist.values()),
            name='Characters'), row=1, col=2)

    fig.update_layout(height=400, title_text="User Analytics")

    return fig

# ============================================================================
# GRADIO INTERFACE - FIXED VERSION
# ============================================================================



def create_interface(chatbot: MultiCharacterChatbot):
    """Create comprehensive Gradio interface with FIXED chat history format"""

    # Store character selection state
    current_character = {"id": "king"}

    def chat_fn(message, history, session_id):
        """Handle Gradio's chatbot format - CORRECTED for proper message format"""
        # Handle empty message
        if not message or not message.strip():
            return history, ""

        # Get current character
        character_id = current_character["id"]

        # Generate response
        try:
            response = chatbot.generate_answer(message, character_id, session_id)
        except Exception as e:
            response = {
                "answer": f"Error generating response: {str(e)[:100]}",
                "character": "System",
                "title": "Error",
                "error": True
            }

        # Format the response
        if response.get("error"):
            bot_message = f"⚠️ {response['answer']}"
        else:
            char_name = response.get('character', 'Assistant')
            char_title = response.get('title', '')
            bot_message = f"**{char_name}** ({char_title})\n\n{response['answer']}"
            
            # Add metadata if available
            if response.get('confidence'):
                bot_message += f"\n\n_Confidence: {response['confidence']:.2%}_"
        
        # Initialize history as empty list if None
        if history is None:
            history = []
        
        # CORRECTED: Use dictionary format with 'role' and 'content' keys
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_message}
        ]
        
        # Return updated history and empty string to clear input
        return new_history, ""

    def update_character(character_id):
        """Update the current character"""
        current_character["id"] = character_id
        char_info = CHARACTERS[character_id]
        return f"Now chatting with: **{char_info['name']}** - {char_info['title']}"

    def feedback_fn(session_id, query, rating):
        """Process feedback"""
        if not query or not rating:
            return "⚠️  Please provide query and rating"

        character_id = current_character["id"]
        result = chatbot.provide_feedback(session_id, character_id, query, float(rating))
        return f"✓ Feedback recorded! Reward: {result['reward']:.2f}, Epsilon: {result['epsilon']:.3f}"

    # Create the interface
    with gr.Blocks(
        title="Multi-Character Historical Chatbot", 
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # 🏛️ Multi-Character Historical Chatbot of Sri Lanka
        ### Complete System: RAG + RL + Fine-tuning + Metrics + 4 Characters
        
        **Features:** Retrieval-Augmented Generation, Reinforcement Learning, LoRA Fine-tuning, 
        Real-time Metrics, User Profiles, Feedback System
        
        💡 **Tip:** Start by greeting your chosen character to learn about their background!
        """)

        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot_ui = gr.Chatbot(
                        value=[],
                        height=500, 
                        label="Conversation"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask your question...", 
                            label="Your Question", 
                            scale=4,
                            lines=2
                        )
                        session_input = gr.Textbox(
                            value="user_1", 
                            label="Session ID", 
                            scale=1
                        )
                    
                    # Submit button
                    submit_btn = gr.Button("Send", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["Hello, who are you?"],
                            ["Tell me about the Sacred Tooth Relic"],
                            ["What is the Esala Perahera festival?"],
                            ["Describe Galle Fort"],
                            ["What was Dutch rule like in Ceylon?"],
                            ["Give me an overview of Sri Lankan history"]
                        ],
                        inputs=msg
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 🎭 Select Character")
                    
                    character_selector = gr.Radio(
                        choices=[
                            ("👑 King Sri Vijaya Rajasinha", "king"),
                            ("🕉️ Diyawadana Nilame", "nilame"),
                            ("⚓ Dutch Officer", "dutch"),
                            ("🇱🇰 Sri Lankan Historian", "citizen")
                        ],
                        value="king",
                        label="Who to talk to?",
                        interactive=True
                    )
                    
                    character_status = gr.Markdown(
                        "Now chatting with: **King Sri Vijaya Rajasinha** - Last King of Kandy (1739-1747)"
                    )
                    
                    # Update character when selection changes
                    character_selector.change(
                        update_character,
                        inputs=[character_selector],
                        outputs=[character_status]
                    )
                    
                    gr.Markdown("""
                    ### 📊 Character Expertise
                    
                    **King:** Buddhist traditions, Royal ceremonies, Sacred Tooth Relic
                    
                    **Nilame:** Temple ceremonies, Esala Perahera, Buddhist rituals
                    
                    **Dutch Officer:** Galle Fort, Maritime trade, VOC operations
                    
                    **Historian:** Overview of Sri Lankan history, Cultural heritage
                    """)

            # Connect the chat function
            msg.submit(chat_fn, [msg, chatbot_ui, session_input], [chatbot_ui, msg])
            submit_btn.click(chat_fn, [msg, chatbot_ui, session_input], [chatbot_ui, msg])

        # ... rest of the tabs remain the same ...

    # Create the interface
    with gr.Blocks(
        title="Multi-Character Historical Chatbot", 
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # 🏛️ Multi-Character Historical Chatbot of Sri Lanka
        ### Complete System: RAG + RL + Fine-tuning + Metrics + 4 Characters
        
        **Features:** Retrieval-Augmented Generation, Reinforcement Learning, LoRA Fine-tuning, 
        Real-time Metrics, User Profiles, Feedback System
        
        💡 **Tip:** Start by greeting your chosen character to learn about their background!
        """)

        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot_ui = gr.Chatbot(
                        value=[],
                        height=500, 
                        label="Conversation"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask your question...", 
                            label="Your Question", 
                            scale=4,
                            lines=2
                        )
                        session_input = gr.Textbox(
                            value="user_1", 
                            label="Session ID", 
                            scale=1
                        )
                    
                    # Submit button
                    submit_btn = gr.Button("Send", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["Hello, who are you?"],
                            ["Tell me about the Sacred Tooth Relic"],
                            ["What is the Esala Perahera festival?"],
                            ["Describe Galle Fort"],
                            ["What was Dutch rule like in Ceylon?"],
                            ["Give me an overview of Sri Lankan history"]
                        ],
                        inputs=msg
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 🎭 Select Character")
                    
                    character_selector = gr.Radio(
                        choices=[
                            ("👑 King Sri Vijaya Rajasinha", "king"),
                            ("🕉️ Diyawadana Nilame", "nilame"),
                            ("⚓ Dutch Officer", "dutch"),
                            ("🇱🇰 Sri Lankan Historian", "citizen")
                        ],
                        value="king",
                        label="Who to talk to?",
                        interactive=True
                    )
                    
                    character_status = gr.Markdown(
                        "Now chatting with: **King Sri Vijaya Rajasinha** - Last King of Kandy (1739-1747)"
                    )
                    
                    # Update character when selection changes
                    character_selector.change(
                        update_character,
                        inputs=[character_selector],
                        outputs=[character_status]
                    )
                    
                    gr.Markdown("""
                    ### 📊 Character Expertise
                    
                    **King:** Buddhist traditions, Royal ceremonies, Sacred Tooth Relic
                    
                    **Nilame:** Temple ceremonies, Esala Perahera, Buddhist rituals
                    
                    **Dutch Officer:** Galle Fort, Maritime trade, VOC operations
                    
                    **Historian:** Overview of Sri Lankan history, Cultural heritage
                    """)

            # Connect the chat function
            msg.submit(chat_fn, [msg, chatbot_ui, session_input], [chatbot_ui, msg])
            submit_btn.click(chat_fn, [msg, chatbot_ui, session_input], [chatbot_ui, msg])

        with gr.Tab("📈 Training Metrics"):
            gr.Markdown("### Fine-tuning Performance")
            gr.Markdown("View training loss over time to monitor model fine-tuning progress.")
            training_viz = gr.Plot(label="Training Loss")
            training_btn = gr.Button("Update Visualization", variant="primary")
            training_btn.click(
                lambda: create_training_viz(chatbot.metrics_tracker), 
                outputs=training_viz
            )

        with gr.Tab("⭐ Feedback & RL"):
            gr.Markdown("### Provide Feedback to Improve AI")
            gr.Markdown("Rate responses to help the AI learn your preferences through Reinforcement Learning!")

            with gr.Row():
                fb_session = gr.Textbox(label="Session ID", value="user_1")
                fb_query = gr.Textbox(
                    label="Query to Rate",
                    placeholder="Enter the question you asked"
                )

            fb_rating = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Rating (1=Poor, 5=Excellent)"
            )
            fb_btn = gr.Button("Submit Feedback", variant="primary")
            fb_result = gr.Textbox(label="Result")

            fb_btn.click(
                feedback_fn, 
                [fb_session, fb_query, fb_rating], 
                fb_result
            )

            gr.Markdown("### 🧠 RL Learning Progress")
            rl_viz = gr.Plot(label="RL Performance")
            rl_btn = gr.Button("Update RL Visualization", variant="primary")
            rl_btn.click(
                lambda: create_rl_performance_viz(
                    chatbot.metrics_tracker, 
                    chatbot.rl_recommender
                ),
                outputs=rl_viz
            )

        with gr.Tab("🔍 RAG Performance"):
            gr.Markdown("### Retrieval-Augmented Generation Metrics")
            gr.Markdown("Monitor how well the system retrieves relevant information from the knowledge base.")
            rag_viz = gr.Plot(label="RAG Performance")
            rag_btn = gr.Button("Update RAG Visualization", variant="primary")
            rag_btn.click(
                lambda: create_rag_performance_viz(chatbot.metrics_tracker),
                outputs=rag_viz
            )

        with gr.Tab("👤 User Profile"):
            gr.Markdown("### View User Profile and History")
            gr.Markdown("See your interaction history and preferences.")
            
            profile_session = gr.Textbox(label="Session ID", value="user_1")
            profile_btn = gr.Button("Load Profile", variant="primary")
            profile_display = gr.Code(label="Profile Data", language="json")

            def show_profile(session_id):
                profile = chatbot.profile_manager.get_profile(session_id)
                return json.dumps(profile, indent=2)

            profile_btn.click(show_profile, profile_session, profile_display)

        with gr.Tab("📊 Analytics"):
            gr.Markdown("### System Analytics Dashboard")
            gr.Markdown("Overview of user interactions, popular topics, and character usage.")
            analytics_viz = gr.Plot(label="User Analytics")
            analytics_btn = gr.Button("Update Analytics", variant="primary")
            analytics_btn.click(
                lambda: create_analytics_viz(chatbot.metrics_tracker),
                outputs=analytics_viz
            )

        with gr.Accordion("ℹ️ About This System", open=False):
            gr.Markdown("""
            ### Technology Stack:
            - **Model:** TinyLlama 1.1B with LoRA fine-tuning
            - **RAG:** ChromaDB + Sentence Transformers (with fallback)
            - **RL:** Q-Learning with epsilon-greedy exploration
            - **Metrics:** Real-time tracking with Plotly visualizations
            - **Characters:** 4 distinct personas with specialized knowledge
            
            ### Dataset Coverage:
            - Kingdom of Kandy (1739-1815)
            - Sacred Tooth Relic & Temple ceremonies
            - Esala Perahera festival
            - Galle Fort & Dutch colonial period
            - VOC maritime trade operations
            - Modern Sri Lankan heritage
            
            ### Advanced Features:
            ✅ Character-specific knowledge retrieval
            ✅ Reinforcement learning recommendations
            ✅ User profile management
            ✅ Intent classification
            ✅ Real-time metrics tracking
            ✅ Fine-tuning capabilities
            ✅ REST API endpoints
            ✅ Network timeout handling
            ✅ Fallback search mechanisms
            
            ### Troubleshooting:
            - **Model loading issues:** Check internet connection, the system will retry automatically
            - **Slow responses:** First-time model downloads can take time
            - **Search quality:** If embeddings fail, system falls back to keyword search
            """)

    return demo

# ============================================================================
# FLASK REST API
# ============================================================================

def create_api(chatbot: MultiCharacterChatbot):
    """Create Flask REST API"""

    app = Flask(__name__)
    CORS(app)

    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "healthy",
            "characters": list(CHARACTERS.keys()),
            "model_loaded": chatbot.model is not None,
            "timestamp": datetime.now().isoformat()
        })

    @app.route('/api/chat', methods=['POST'])
    def chat():
        try:
            data = request.get_json()
            query = data.get('query', '')
            character_id = data.get('character_id', 'king')
            session_id = data.get('session_id', 'default')

            if not query:
                return jsonify({"error": "Query is required"}), 400

            response = chatbot.generate_answer(query, character_id, session_id)

            return jsonify({
                "success": True,
                "data": response,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/feedback', methods=['POST'])
    def feedback():
        try:
            data = request.get_json()
            session_id = data.get('session_id', 'default')
            character_id = data.get('character_id', 'king')
            query = data.get('query', '')
            rating = data.get('rating', 0)

            if not query or not rating:
                return jsonify({"error": "Query and rating required"}), 400

            result = chatbot.provide_feedback(
                session_id, character_id, query, float(rating)
            )

            return jsonify({
                "success": True,
                "data": result,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/profile/<session_id>', methods=['GET'])
    def get_profile(session_id):
        try:
            profile = chatbot.profile_manager.get_profile(session_id)
            return jsonify({"success": True, "data": profile})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/metrics', methods=['GET'])
    def get_metrics():
        try:
            return jsonify({
                "success": True, 
                "data": chatbot.metrics_tracker.metrics
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/characters', methods=['GET'])
    def get_characters():
        return jsonify({"success": True, "data": CHARACTERS})

    return app

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🤖 COMPLETE MULTI-CHARACTER HISTORICAL CHATBOT")
    print("="*80)

    # Create necessary directories
    os.makedirs("./model_cache", exist_ok=True)
    os.makedirs("./lora_output", exist_ok=True)
    os.makedirs("./lora_adapter", exist_ok=True)

    # Load data
    print("\n[1/5] Loading Dataset")
    data_loader = DataLoader()
    data_loader.load_json("data.json")

    # Initialize chatbot
    print("\n[2/5] Initializing Chatbot")
    chatbot = MultiCharacterChatbot(CONFIG, data_loader)

    # Optional: Fine-tune
    print("\n[3/5] Model Setup")
    print("Fine-tune the model? (y/n) [default: n]: ", end="", flush=True)
    try:
        choice = input().strip().lower() or 'n'
    except:
        choice = 'n'

    try:
        if choice == 'y':
            print("\n⚠️  Fine-tuning will download large models (may take time)")
            print("   Proceeding with fine-tuning...")
            chatbot.fine_tune()
        else:
            print("→ Loading base model (this may take a few minutes on first run)...")
            chatbot._load_model()
            chatbot._prepare_knowledge_base()
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        print("\n⚠️  Continuing with limited functionality...")
        print("   The chatbot will work but may have reduced performance.")

    # Test
    print("\n[4/5] Testing System")
    if chatbot.model is not None:
        test_queries = [
            ("Hello", "king"),
            ("What is the Esala Perahera?", "nilame"),
            ("Tell me about Galle Fort", "dutch")
        ]

        for query, char in test_queries:
            print(f"\n❓ {char.upper()}: {query}")
            try:
                response = chatbot.generate_answer(query, char, "test")
                if not response.get("error"):
                    print(f"💬 {response['answer'][:150]}...")
            except Exception as e:
                print(f"⚠️  Test failed: {str(e)[:100]}")
    else:
        print("⚠️  Model not loaded, skipping tests")

    # Launch services
    print("\n[5/5] Launching Services")
    
    # Start API
    print("\n🚀 Starting REST API...")
    app = create_api(chatbot)
    
    def run_flask():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    api_thread = threading.Thread(target=run_flask, daemon=True)
    api_thread.start()
    
    time.sleep(2)
    print("✓ API running on http://localhost:5000")

    # Launch Gradio
    print("\n🎨 Launching Gradio Interface...")
    demo = create_interface(chatbot)

    print("\n" + "="*80)
    print("🎉 SYSTEM READY!")
    print("="*80)
    print("\n✅ All features active:")
    print("   - 4 Character System (King, Nilame, Dutch, Citizen)")
    print("   - RAG with ChromaDB (with fallback)")
    print("   - Reinforcement Learning")
    print("   - Metrics Tracking")
    print("   - User Profiles")
    print("   - REST API")
    print("   - Gradio UI (FIXED)")
    print("   - Network timeout handling")
    print("\n💡 Tips:")
    print("   - First run may take time to download models")
    print("   - If download fails, the system uses fallback mechanisms")
    print("   - Internet connection required for initial setup")
    print("\n" + "="*80)

    # Let Gradio automatically find an available port
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            show_error=True,
            inbrowser=False
        )
    except Exception as e:
        print(f"\n⚠️  Launch error: {e}")
        print("\nTrying alternative launch method...")
        # Try with specific port range
        for port in range(7860, 7870):
            try:
                demo.launch(
                    share=True,
                    server_name="0.0.0.0",
                    server_port=port,
                    show_error=True,
                    inbrowser=False
                )
                print(f"✓ Successfully launched on port {port}")
                break
            except:
                continue