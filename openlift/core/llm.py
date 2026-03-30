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

try:
    import openai
except ImportError:
    openai = None

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
        elif self.provider == "deepseek":
            self._init_deepseek()

    def _init_deepseek(self):
        key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if key and openai:
            try:
                self.client = openai.OpenAI(api_key=key, base_url="https://api.deepseek.com")
                self.model_name = self.model_name or "deepseek-reasoner"
            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek client: {e}")

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
        # Default to 127.0.0.1 to avoid IPv6 localhost issues (Errno 99)
        self.ollama_base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model_name = self.model_name or "llama3"

    def is_available(self) -> bool:
        if self.provider in ["gemini", "deepseek"]:
            return self.client is not None
        elif self.provider == "ollama":
            return True # Assumed available, wil fail at request time if not
        return False

    def _call_ollama(self, prompt: str) -> str:
        """
        Make a raw REST request to Ollama /api/generate
        """
        urls_to_try = [f"{self.ollama_base_url}/api/generate"]
        
        # Fallback logic: if default is 127.0.0.1, try localhost, and vice-versa
        if "127.0.0.1" in self.ollama_base_url:
            urls_to_try.append(self.ollama_base_url.replace("127.0.0.1", "localhost") + "/api/generate")
        elif "localhost" in self.ollama_base_url:
             urls_to_try.append(self.ollama_base_url.replace("localhost", "127.0.0.1") + "/api/generate")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        data = json.dumps(payload).encode("utf-8")
        
        last_error = None
        
        for url in urls_to_try:
            try:
                # logger.info(f"Trying Ollama at {url}")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return result.get("response", "").strip()
            except urllib.error.URLError as e:
                last_error = e
                # Continue to next URL
                continue
            except Exception as e:
                return f"❌ Ollama Error: {e}"
        
        return f"❌ Ollama Connection Error: {last_error}. Is Ollama running?"

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
            res = self._call_ollama(prompt)
            return {"content": res, "reasoning": None}
        elif self.provider == "deepseek":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                msg = response.choices[0].message
                return {
                    "content": msg.content.strip() if msg.content else "Done",
                    "reasoning": getattr(msg, 'reasoning_content', None)
                }
            except Exception as e:
                return {"content": f"Error generating insights with DeepSeek: {e}", "reasoning": None}
        
        # Default Gemini
        try:
            response = self.client.generate_content(prompt)
            return {"content": response.text.strip(), "reasoning": None}
        except Exception as e:
            return {"content": f"Error generating insights with Gemini: {e}", "reasoning": None}

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
            res = self._call_ollama(prompt)
            return {"content": res, "reasoning": None}
        elif self.provider == "deepseek":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                msg = response.choices[0].message
                return {
                    "content": msg.content.strip() if msg.content else "Done",
                    "reasoning": getattr(msg, 'reasoning_content', None)
                }
            except Exception as e:
                return {"content": f"Error generating insights with DeepSeek: {e}", "reasoning": None}
            
        # Default Gemini
        try:
            response = self.client.generate_content(prompt)
            return {"content": response.text.strip(), "reasoning": None}
        except Exception as e:
            return {"content": f"Error generating insights with Gemini: {e}", "reasoning": None}

    def get_multi_cell_insights(self, multi_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate cross-channel strategy insights for a multi-cell experiment.
        """
        if not self.is_available():
            return {"content": f"{self.provider.title()} insights unavailable.", "reasoning": None}

        cells = multi_results.get("cells", {})
        comparisons = multi_results.get("comparisons", {})
        synergy = multi_results.get("synergy")

        # Build cell summary
        cell_lines = []
        for label, result in cells.items():
            if "error" in result:
                cell_lines.append(f"- {label}: FAILED ({result['error']})")
                continue
            m = result.get("metrics", {})
            cell_lines.append(
                f"- {label} ({result.get('cell_name', '')}): "
                f"Lift = {m.get('incremental_outcome_mean', 0):.1f}, "
                f"Lift% = {m.get('lift_pct_mean', 0):.1f}%, "
                f"Confidence = {m.get('p_positive', 0)*100:.0f}%"
            )
        cell_summary = "\n".join(cell_lines)

        # Build comparison summary
        comp_lines = []
        for pair, comp in comparisons.items():
            if "error" in comp:
                comp_lines.append(f"- {pair}: Error")
                continue
            winner = comp.get("winner", "Inconclusive")
            confidence = comp.get("confidence_level", "low")
            comp_lines.append(f"- {pair}: Winner = {winner} ({confidence} confidence)")
        comp_summary = "\n".join(comp_lines)

        # Build synergy summary
        synergy_text = "No synergy analysis available."
        if synergy:
            synergy_text = (
                f"Combined cell '{synergy['combined_cell']}' lift = {synergy['combined_lift']:.1f}, "
                f"sum of individual lifts = {synergy['sum_individual_lifts']:.1f}, "
                f"synergy delta = {synergy['synergy_delta']:.1f} ({synergy['synergy_pct']:.1f}%), "
                f"super-additive = {synergy['is_super_additive']}"
            )

        prompt = f"""
        You are a senior Marketing Analyst and Media Strategist.
        Analyze the results of this Multi-Cell Geo-Lift experiment and provide
        a strategic cross-channel recommendation for a CMO.

        **Per-Cell Results:**
        {cell_summary}

        **Pairwise Comparisons:**
        {comp_summary}

        **Synergy Analysis:**
        {synergy_text}

        **Instructions:**
        1. Rank the channels/cells from best to worst performer.
        2. For each channel, state whether to SCALE, MAINTAIN, or CUT budget.
        3. If synergy exists, explain why the combined approach works.
        4. Give a clear budget reallocation recommendation (e.g., "shift 30% of Cell B budget to Cell A").
        5. Mention confidence levels — be honest about statistical certainty.
        6. Keep it under 200 words. Use formatting for executive readability.
        """

        if self.provider == "ollama":
            res = self._call_ollama(prompt)
            return {"content": res, "reasoning": None}
        elif self.provider == "deepseek":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                msg = response.choices[0].message
                return {
                    "content": msg.content.strip() if msg.content else "Done",
                    "reasoning": getattr(msg, 'reasoning_content', None)
                }
            except Exception as e:
                return {"content": f"Error: {e}", "reasoning": None}

        # Default Gemini
        try:
            response = self.client.generate_content(prompt)
            return {"content": response.text.strip(), "reasoning": None}
        except Exception as e:
            return {"content": f"Error: {e}", "reasoning": None}
