import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
import logging

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "ollama":
            self._init_ollama()

    def _init_gemini(self):
        # Prefer provided key, fallback to env var
        key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if key and genai:
            try:
                genai.configure(api_key=key)
                # Use provided model or default to 2.0-flash
                model = self.model_name or 'gemini-2.0-flash'
                self.client = genai.GenerativeModel(model)
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")

    def _init_ollama(self):
        # No client needed for REST calls, just set base URL
        self.ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = self.model_name or "llama3"

    def is_available(self) -> bool:
        if self.provider == "gemini":
            return self.client is not None
        elif self.provider == "ollama":
            return True # Assumed available, wil fail at request time if not
        return False

    def _call_ollama(self, prompt: str) -> str:
        """
        Make a raw REST request to Ollama /api/generate
        """
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        data = json.dumps(payload).encode("utf-8")
        
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"❌ Ollama Connection Error: {e}. Is Ollama running?"
        except Exception as e:
            return f"❌ Ollama Error: {e}"

    def get_experiment_insights(self, experiment_results: Dict[str, Any], context_str: str = "") -> str:
        """
        Generate insights using the selected provider.
        """
        if not self.is_available():
            return f"{self.provider.title()} insights unavailable. Check configuration."

        m = experiment_results.get('metrics', {})
        
        prompt = f"""
        You are a senior Data Scientist and Marketing Analyst expert.
        Analyze the results of this Geo-Lift experiment and provide a concise, executive summary for a marketing stakeholder.
        
        **Experiment Context:**
        {context_str}
        
        **Key Metrics:**
        - Incremental Lift: {m.get('incremental_outcome_mean', 0):.2f}
        - Lift Percentage: {m.get('lift_pct_mean', 0):.2f}%
        - Probability of Positive Lift: {m.get('p_positive', 0)*100:.1f}%
        
        **Instructions:**
        1. Start with a direct "Verdict": Was the test successful, inconclusive, or negative?
        2. Explain the "Lift" in plain English (e.g., "We drove X extra conversions...").
        3. Explain confidence/significance naturally (e.g., "We are 95% sure...").
        4. If ROI data is implied (incremental > 0), mention efficiency.
        5. Keep it strictly under 150 words. Use formatting (bolding) for impact.
        """
        
        if self.provider == "ollama":
            return self._call_ollama(prompt)
        
        # Default Gemini
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating insights with Gemini: {e}"

    def get_power_analysis_insights(self, power_results: Dict[str, Any]) -> str:
        """
        Generate insights using the selected provider.
        """
        if not self.is_available():
            return f"{self.provider.title()} insights unavailable."

        power = power_results.get('power', 0)
        sims = power_results.get('simulations', 0)
        lift = power_results.get('effect_size', 0)
        duration = power_results.get('duration', 0)
        
        prompt = f"""
        You are a senior Data Scientist.
        Interpret this Power Analysis simulation for a user planning a marketing test.
        
        **Config:**
        - Test Duration: {duration} days
        - Expected/Target Lift: {lift*100:.1f}%
        - Simulations Run: {sims}
        
        **Result:**
        - Estimated Power (Probability of Detection): {power*100:.1f}%
        
        **Instructions:**
        1. Give a "Go/No-Go" recommendation.
        2. Explain what the Power means in this specific context.
        3. If Power is low (<80%), suggest specific actions.
        4. Keep it under 100 words.
        """
        
        if self.provider == "ollama":
            return self._call_ollama(prompt)
            
        # Default Gemini
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating insights with Gemini: {e}"
