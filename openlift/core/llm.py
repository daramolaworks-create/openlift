import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Optional provider SDKs — imported lazily so the core library works without them.
try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic as anthropic_sdk
except ImportError:
    anthropic_sdk = None


class LLMService:
    """
    Unified LLM gateway supporting Gemini, Claude (Anthropic), DeepSeek, and Ollama.

    All public ``get_*`` methods return ``{"content": str, "reasoning": str | None}``.
    """

    PROVIDERS = ("gemini", "claude", "deepseek", "ollama")

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.client = None

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "claude":
            self._init_claude()
        elif self.provider == "deepseek":
            self._init_deepseek()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            logger.warning(f"Unknown LLM provider: {provider!r}. Supported: {self.PROVIDERS}")

    # ------------------------------------------------------------------
    # Provider initialisation
    # ------------------------------------------------------------------

    def _init_gemini(self):
        key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key or not google_genai:
            return
        try:
            self.client = google_genai.Client(api_key=key)
            self.model_name = self.model_name or "gemini-2.5-flash"
        except Exception as e:
            logger.error(f"Failed to initialise Gemini client: {e}")

    def _init_claude(self):
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key or not anthropic_sdk:
            return
        try:
            self.client = anthropic_sdk.Anthropic(api_key=key)
            self.model_name = self.model_name or "claude-sonnet-4-6"
        except Exception as e:
            logger.error(f"Failed to initialise Anthropic client: {e}")

    def _init_deepseek(self):
        key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key or not openai:
            return
        try:
            self.client = openai.OpenAI(api_key=key, base_url="https://api.deepseek.com")
            self.model_name = self.model_name or "deepseek-reasoner"
        except Exception as e:
            logger.error(f"Failed to initialise DeepSeek client: {e}")

    def _init_ollama(self):
        # Default to 127.0.0.1 to avoid IPv6 issues (Errno 99).
        self.ollama_base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model_name = self.model_name or "llama3"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self.provider == "ollama":
            return True  # Assumed running; will fail at request time if not.
        return self.client is not None

    # ------------------------------------------------------------------
    # Core dispatch — every public method funnels through here
    # ------------------------------------------------------------------

    def _dispatch(self, prompt: str) -> Dict[str, Any]:
        """Send *prompt* to the configured provider and return a normalised result dict."""
        if self.provider == "ollama":
            return {"content": self._call_ollama(prompt), "reasoning": None}

        if self.provider == "deepseek":
            return self._call_openai_compat(prompt)

        if self.provider == "claude":
            return self._call_claude(prompt)

        # Default: Gemini
        return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return {"content": response.text.strip(), "reasoning": None}
        except Exception as e:
            return {"content": f"Gemini error: {e}", "reasoning": None}

    def _call_claude(self, prompt: str) -> Dict[str, Any]:
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return {"content": message.content[0].text.strip(), "reasoning": None}
        except Exception as e:
            return {"content": f"Claude error: {e}", "reasoning": None}

    def _call_openai_compat(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = response.choices[0].message
            return {
                "content": msg.content.strip() if msg.content else "",
                "reasoning": getattr(msg, "reasoning_content", None),
            }
        except Exception as e:
            return {"content": f"{self.provider.title()} error: {e}", "reasoning": None}

    def _call_ollama(self, prompt: str) -> str:
        urls_to_try = [f"{self.ollama_base_url}/api/generate"]
        if "127.0.0.1" in self.ollama_base_url:
            urls_to_try.append(
                self.ollama_base_url.replace("127.0.0.1", "localhost") + "/api/generate"
            )
        elif "localhost" in self.ollama_base_url:
            urls_to_try.append(
                self.ollama_base_url.replace("localhost", "127.0.0.1") + "/api/generate"
            )

        payload = json.dumps({"model": self.model_name, "prompt": prompt, "stream": False}).encode()
        last_error = None
        for url in urls_to_try:
            try:
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req) as resp:
                    return json.loads(resp.read().decode()).get("response", "").strip()
            except urllib.error.URLError as e:
                last_error = e
            except Exception as e:
                return f"Ollama error: {e}"
        return f"Ollama connection error: {last_error}. Is Ollama running?"

    # ------------------------------------------------------------------
    # Public insight methods
    # ------------------------------------------------------------------

    def get_experiment_insights(
        self, experiment_results: Dict[str, Any], context_str: str = ""
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {"content": f"{self.provider.title()} unavailable. Check configuration.", "reasoning": None}

        m = experiment_results.get("metrics", {})
        prompt = f"""You are a senior Data Scientist and Marketing Analyst.
Analyse the results of this Geo-Lift experiment and provide a concise executive summary for a marketing stakeholder.

**Experiment Context:**
{context_str}

**Key Metrics:**
- Incremental Lift: {m.get('incremental_outcome_mean', 0):.2f}
- Lift Percentage: {m.get('lift_pct_mean', 0):.2f}%
- Probability of Positive Lift: {m.get('p_positive', 0) * 100:.1f}%

**Instructions:**
1. Start with a direct "Verdict": Was the test successful, inconclusive, or negative?
2. Explain the "Lift" in plain English (e.g., "We drove X extra conversions...").
3. Explain confidence naturally (e.g., "We are 95% confident...").
4. If ROI is implied (incremental > 0), mention efficiency.
5. Keep strictly under 150 words. Use **bold** for impact."""
        return self._dispatch(prompt)

    def get_power_analysis_insights(self, power_results: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            return {"content": f"{self.provider.title()} unavailable.", "reasoning": None}

        prompt = f"""You are a senior Data Scientist.
Interpret this Power Analysis simulation for a user planning a marketing test.

**Config:**
- Test Duration: {power_results.get('duration', 0)} days
- Expected/Target Lift: {power_results.get('effect_size', 0) * 100:.1f}%
- Simulations Run: {power_results.get('simulations', 0)}

**Result:**
- Estimated Power (Probability of Detection): {power_results.get('power', 0) * 100:.1f}%

**Instructions:**
1. Give a "Go/No-Go" recommendation.
2. Explain what the Power means in this specific context.
3. If Power is low (<80%), suggest specific actions.
4. Keep under 100 words."""
        return self._dispatch(prompt)

    def get_multi_cell_insights(self, multi_results: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            return {"content": f"{self.provider.title()} unavailable.", "reasoning": None}

        cells = multi_results.get("cells", {})
        comparisons = multi_results.get("comparisons", {})
        synergy = multi_results.get("synergy")

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
                f"Confidence = {m.get('p_positive', 0) * 100:.0f}%"
            )

        comp_lines = []
        for pair, comp in comparisons.items():
            if "error" in comp:
                comp_lines.append(f"- {pair}: Error")
                continue
            comp_lines.append(
                f"- {pair}: Winner = {comp.get('winner', 'Inconclusive')} "
                f"({comp.get('confidence_level', 'low')} confidence)"
            )

        synergy_text = "No synergy analysis available."
        if synergy:
            synergy_text = (
                f"Combined cell '{synergy['combined_cell']}' lift = {synergy['combined_lift']:.1f}, "
                f"sum of individual lifts = {synergy['sum_individual_lifts']:.1f}, "
                f"synergy delta = {synergy['synergy_delta']:.1f} ({synergy['synergy_pct']:.1f}%), "
                f"super-additive = {synergy['is_super_additive']}"
            )

        prompt = f"""You are a senior Marketing Analyst and Media Strategist.
Analyse the results of this Multi-Cell Geo-Lift experiment and provide a strategic cross-channel recommendation for a CMO.

**Per-Cell Results:**
{chr(10).join(cell_lines)}

**Pairwise Comparisons:**
{chr(10).join(comp_lines)}

**Synergy Analysis:**
{synergy_text}

**Instructions:**
1. Rank channels from best to worst performer.
2. For each, state whether to SCALE, MAINTAIN, or CUT budget.
3. If synergy exists, explain why the combined approach works.
4. Give a clear budget reallocation recommendation.
5. Mention confidence levels honestly.
6. Keep under 200 words. Use **bold** for executive readability."""
        return self._dispatch(prompt)
