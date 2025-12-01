
import os
import json
import logging
from typing import Any, Dict, List, Optional

import requests

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """
    Thin wrapper around OpenRouter chat API for BMW used-car analytics.

    Typical usage:
        orchestrator = LLMOrchestrator()
        explanation = orchestrator.explain_chart(chart_summary)
        full_report = orchestrator.full_market_analysis(list_of_summaries)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "x-ai/grok-4.1-fast:free",
        timeout: int = 30,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        default_title: str = "bmw-used-car-analytics",
    ) -> None:
        
        """
        Initialize the orchestrator.

        Args:
            api_key: OpenRouter API key. If None, tries env var OPENROUTER_API_KEY.
            model: Default model identifier on OpenRouter.
            timeout: HTTP request timeout in seconds.
            base_url: OpenRouter chat completions endpoint.
            default_title: Default X-Title header for OpenRouter.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Please provide api_key or env var.")

        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.default_title = default_title

    # ---- 1. Low-level HTTP call wrapper ----
    def _call_openrouter(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        title: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Internal helper to call OpenRouter chat API.

        Args:
            messages: OpenAI-style message list.
            model: Optional override of default model.
            title: Optional request title for OpenRouter.
            temperature: Sampling temperature.
            max_tokens: Optional max tokens for completion.

        Returns:
            str: Assistant message content.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": title or self.default_title,
        }

        try:
            resp = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.error("Request to OpenRouter failed", exc_info=True)
            raise RuntimeError(f"Request to OpenRouter failed: {e}") from e

        if resp.status_code != 200:
            logger.error(
                "OpenRouter API error: status=%s, body=%s",
                resp.status_code,
                resp.text,
            )
            raise RuntimeError(
                f"OpenRouter API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error("Unexpected OpenRouter response structure: %s", data, exc_info=True)
            raise RuntimeError(f"Unexpected OpenRouter response structure: {data}") from e
        
    # ---- 2. High-level Business Logic (Map Step) ----
    def generate_chart_narrative(
        self, 
        section_title: str, 
        data_summary: Any, 
        specific_instruction: str = ""
    ) -> str:
        """
        Generates a specific section analysis based on a chart and statistical summary.
        The output is pure text analysis, without any Markdown headings or image links.
        """
        
        # 将复杂对象转为字符串
        summary_str = json.dumps(data_summary, indent=2, default=str) if not isinstance(data_summary, str) else data_summary

        system_prompt = (
            "You are a Senior Data Analyst at BMW, specializing in the **global used-car market**. "
            "Your task is to write a concise, professional analysis for a business report section. "
            "Output MUST be plain text, do NOT use any Markdown headings, bolding, lists, or image links."
        )

        user_prompt = f"""
        ### Task
        Write the narrative analysis for the **{section_title}** section.

        ### Context
        - The analysis will be inserted into a sub-section titled: {section_title}
        
        ### Data Summary (The Source of Truth)
        Use the following statistics to support your analysis:
        {summary_str}

        ### Instructions (Optimized for Quantitative Analysis)
        1.  **STRICT FORMATTING:** DO NOT output any image links, Markdown headings, bolding, or list items. The output must be pure, continuous text.
        2.  **PRIORITIZE QUANTIFICATION:** Focus the analysis on the derived quantitative metrics provided (e.g., **CAGR, Market Share Shift, Concentration Ratios, Disparity Ratios**). Do not simply describe the chart; interpret the business meaning of these metrics.
        3.  **INSIGHT GENERATION:** Start with the most critical insight (e.g., the biggest growth or highest concentration). Explain the implication of the findings for BMW's product strategy, inventory, or **long-term residual value**.
        """

        return self._call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 
        )

    # ---- 3. High-level Business Logic (Reduce Step) ----
    def generate_executive_summary(self, all_narratives: List[str]) -> str:
        """
        Synthesizes all individual section narratives into a cohesive Executive Summary.
        """
        
        # 拼接所有的分析段落作为上下文
        joined_context = "\n\n".join(all_narratives)

        system_prompt = (
            "You are the Chief Strategy Officer at BMW. "
            "Your goal is to synthesize a high-level Executive Summary based on detailed technical reports."
        )

        user_prompt = f"""
        ### Context
        Here are the detailed findings from our data analysis team covering various aspects of sales performance:
        
        {joined_context}

        ### Task
        Write an **Executive Summary** (approx. 300-400 words) with the following structure:
        
        1.  **Overview**: A brief statement on the overall dataset scope and market health.
        2.  **Key Findings**: Summarize the 3 most important trends identified across the reports.
        3.  **Strategic Recommendations**: Provide 3 concrete, actionable business recommendations based on the data (e.g., pricing adjustments, inventory shifts)[cite: 10, 18].
        
        Do not repeat the detailed numbers unnecessarily; focus on the "So What?".
        """

        return self._call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3 # 稍微增加创造性以生成更好的建议
        )