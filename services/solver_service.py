import math
import re
from statistics import NormalDist


class SolverService:
    """Deterministic solver for common statistics numericals."""

    NUMBER_PATTERN = r"[-+]?\d*\.?\d+"

    def __init__(self):
        self.standard_normal = NormalDist(mu=0, sigma=1)

    def _extract_number(self, text, patterns):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_int(self, text, patterns):
        value = self._extract_number(text, patterns)
        if value is None:
            return None
        if float(value).is_integer():
            return int(value)
        return None

    def _extract_data_points(self, question):
        patterns = (
            r"(?:data|values|observations|sample)\s*(?::|=)?\s*([-\d\.,\s]+)",
            r"given\s+the\s+(?:data|values|observations|sample)\s*(?::|=)?\s*([-\d\.,\s]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if not match:
                continue

            chunk = match.group(1)
            numbers = re.findall(self.NUMBER_PATTERN, chunk)
            if len(numbers) >= 2:
                return [float(number) for number in numbers]
        return []

    def _format_number(self, value):
        if value is None:
            return ""
        rounded = round(value, 6)
        if float(rounded).is_integer():
            return str(int(rounded))
        return f"{rounded:.6f}".rstrip("0").rstrip(".")

    def _append_follow_up(self, response, follow_up_prompt):
        return f"{response}\n\n---\n\n{follow_up_prompt}"

    def _build_given_from_data(self, data_points):
        formatted = ", ".join(self._format_number(point) for point in data_points)
        return f"- Data points: \\({formatted}\\)"

    def _extract_interval_bounds(self, question):
        patterns = (
            rf"between\s*({self.NUMBER_PATTERN})\s*and\s*({self.NUMBER_PATTERN})",
            rf"from\s*({self.NUMBER_PATTERN})\s*to\s*({self.NUMBER_PATTERN})",
        )
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return float(match.group(1)), float(match.group(2))
        return None

    def _solve_z_score(self, question, follow_up_prompt):
        lowered = question.lower()
        if "z-score" not in lowered and "z score" not in lowered:
            return None

        x_value = self._extract_number(
            question,
            [
                rf"z[- ]score\s+(?:for|of)\s*\(?({self.NUMBER_PATTERN})\)?",
                rf"for\s*\(?({self.NUMBER_PATTERN})\)?",
                rf"value\s*(?:=|is)?\s*\(?({self.NUMBER_PATTERN})\)?",
                rf"x\s*(?:=|is)?\s*\(?({self.NUMBER_PATTERN})\)?",
            ],
        )
        mean = self._extract_number(
            question,
            [
                rf"mean\s*(?:=|is)?\s*\(?({self.NUMBER_PATTERN})\)?",
                rf"\bmu\b\s*(?:=|is)?\s*\(?({self.NUMBER_PATTERN})\)?",
            ],
        )
        sigma = self._extract_number(
            question,
            [
                rf"(?:sigma|standard deviation|std\.?\s*dev\.?)\s*(?:=|is)?\s*\(?({self.NUMBER_PATTERN})\)?",
            ],
        )

        if x_value is None or mean is None or sigma is None or sigma == 0:
            return None

        z_value = (x_value - mean) / sigma
        response = (
            "## Given\n\n"
            f"- Observation: \\(x = {self._format_number(x_value)}\\)\n"
            f"- Mean: \\(\\mu = {self._format_number(mean)}\\)\n"
            f"- Standard deviation: \\(\\sigma = {self._format_number(sigma)}\\)\n\n"
            "## Formula Used\n\n"
            "$$ z = \\frac{x - \\mu}{\\sigma} $$\n\n"
            "## Step-by-step Solution\n\n"
            f"Substitute the values into the z-score formula:\n\n"
            f"$$ z = \\frac{{{self._format_number(x_value)} - {self._format_number(mean)}}}{{{self._format_number(sigma)}}} $$\n\n"
            f"$$ z = \\frac{{{self._format_number(x_value - mean)}}}{{{self._format_number(sigma)}}} = {self._format_number(z_value)} $$\n\n"
            "## Final Answer\n\n"
            f"**The z-score is \\({self._format_number(z_value)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_sample_mean(self, question, follow_up_prompt):
        lowered = question.lower()
        if "mean" not in lowered and "average" not in lowered:
            return None
        if "variance" in lowered or "standard deviation" in lowered or "z-score" in lowered:
            return None

        data_points = self._extract_data_points(question)
        if len(data_points) < 2:
            return None

        total = sum(data_points)
        count = len(data_points)
        mean_value = total / count
        formatted_sum = " + ".join(self._format_number(point) for point in data_points)
        response = (
            "## Given\n\n"
            f"{self._build_given_from_data(data_points)}\n"
            f"- Number of observations: \\(n = {count}\\)\n\n"
            "## Formula Used\n\n"
            "$$ \\bar{x} = \\frac{\\sum x_i}{n} $$\n\n"
            "## Step-by-step Solution\n\n"
            "First compute the total of all observations:\n\n"
            f"$$ \\sum x_i = {formatted_sum} = {self._format_number(total)} $$\n\n"
            "Now divide by the number of observations:\n\n"
            f"$$ \\bar{{x}} = \\frac{{{self._format_number(total)}}}{{{count}}} = {self._format_number(mean_value)} $$\n\n"
            "## Final Answer\n\n"
            f"**The sample mean is \\(\\bar{{x}} = {self._format_number(mean_value)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_variance_or_std(self, question, follow_up_prompt):
        lowered = question.lower()
        asks_variance = "variance" in lowered
        asks_std = "standard deviation" in lowered or "std dev" in lowered or "std. dev" in lowered
        if not asks_variance and not asks_std:
            return None

        data_points = self._extract_data_points(question)
        if len(data_points) < 2:
            return None

        use_population = "population" in lowered
        n_value = len(data_points)
        mean_value = sum(data_points) / n_value
        squared_terms = [(point - mean_value) ** 2 for point in data_points]
        squared_sum = sum(squared_terms)
        denominator = n_value if use_population else (n_value - 1)
        if denominator <= 0:
            return None

        variance = squared_sum / denominator
        std_dev = math.sqrt(variance)
        divisor_label = "n" if use_population else "n-1"
        variance_symbol = "\\sigma^2" if use_population else "s^2"
        std_symbol = "\\sigma" if use_population else "s"
        squared_expansion = " + ".join(
            f"({self._format_number(point)} - {self._format_number(mean_value)})^2"
            for point in data_points
        )

        final_line = (
            f"**The {'population' if use_population else 'sample'} variance is "
            f"\\({variance_symbol} = {self._format_number(variance)}\\).**"
        )
        if asks_std:
            final_line = (
                f"**The {'population' if use_population else 'sample'} standard deviation is "
                f"\\({std_symbol} = {self._format_number(std_dev)}\\).**"
            )

        response = (
            "## Given\n\n"
            f"{self._build_given_from_data(data_points)}\n"
            f"- Number of observations: \\(n = {n_value}\\)\n\n"
            "## Formula Used\n\n"
            f"$$ {variance_symbol} = \\frac{{\\sum (x_i - \\bar{{x}})^2}}{{{divisor_label}}} $$\n\n"
            f"$$ {std_symbol} = \\sqrt{{{variance_symbol}}} $$\n\n"
            "## Step-by-step Solution\n\n"
            f"First compute the mean:\n\n$$ \\bar{{x}} = {self._format_number(mean_value)} $$\n\n"
            "Now compute the sum of squared deviations:\n\n"
            f"$$ \\sum (x_i - \\bar{{x}})^2 = {squared_expansion} = {self._format_number(squared_sum)} $$\n\n"
            f"Then divide by \\({divisor_label}\\):\n\n"
            f"$$ {variance_symbol} = \\frac{{{self._format_number(squared_sum)}}}{{{denominator}}} = {self._format_number(variance)} $$\n\n"
            f"Finally, take the square root:\n\n"
            f"$$ {std_symbol} = \\sqrt{{{self._format_number(variance)}}} = {self._format_number(std_dev)} $$\n\n"
            "## Final Answer\n\n"
            f"{final_line}"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_binomial_probability(self, question, follow_up_prompt):
        lowered = question.lower()
        if "binomial" not in lowered and "p(x" not in lowered and "probability" not in lowered:
            return None

        k_match = re.search(r"p\s*\(\s*x\s*=\s*(\d+)\s*\)", question, re.IGNORECASE)
        if not k_match:
            k_match = re.search(r"x\s*=\s*(\d+)", question, re.IGNORECASE)
        if not k_match:
            return None

        k_value = int(k_match.group(1))
        n_value = self._extract_int(
            question,
            [
                rf"\bn\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"trials?\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        p_value = self._extract_number(
            question,
            [
                rf"\bp\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"probability\s+of\s+success\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        if n_value is None or p_value is None:
            return None

        if not (0 <= k_value <= n_value and 0 <= p_value <= 1):
            return None

        combination = math.comb(n_value, k_value)
        probability = combination * (p_value ** k_value) * ((1 - p_value) ** (n_value - k_value))
        response = (
            "## Given\n\n"
            f"- Number of trials: \\(n = {n_value}\\)\n"
            f"- Number of successes: \\(x = {k_value}\\)\n"
            f"- Probability of success: \\(p = {self._format_number(p_value)}\\)\n\n"
            "## Formula Used\n\n"
            "$$ P(X = x) = \\binom{n}{x} p^x (1-p)^{n-x} $$\n\n"
            "## Step-by-step Solution\n\n"
            "First compute the combination term:\n\n"
            f"$$ \\binom{{{n_value}}}{{{k_value}}} = {combination} $$\n\n"
            "Now substitute into the binomial formula:\n\n"
            f"$$ P(X = {k_value}) = {combination} \\times ({self._format_number(p_value)})^{k_value} \\times (1-{self._format_number(p_value)})^{{{n_value-k_value}}} $$\n\n"
            f"$$ P(X = {k_value}) = {self._format_number(probability)} $$\n\n"
            "## Final Answer\n\n"
            f"**The required binomial probability is \\({self._format_number(probability)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_poisson_probability(self, question, follow_up_prompt):
        lowered = question.lower()
        if "poisson" not in lowered and "lambda" not in lowered:
            return None

        x_value = self._extract_int(
            question,
            [
                r"p\s*\(\s*x\s*=\s*(\d+)\s*\)",
                r"x\s*=\s*(\d+)",
                r"exactly\s+(\d+)",
            ],
        )
        lambda_value = self._extract_number(
            question,
            [
                rf"lambda\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"mean\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"rate\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        if x_value is None or lambda_value is None or lambda_value <= 0:
            return None

        probability = math.exp(-lambda_value) * (lambda_value ** x_value) / math.factorial(x_value)
        response = (
            "## Given\n\n"
            f"- Count value: \\(x = {x_value}\\)\n"
            f"- Poisson parameter: \\(\\lambda = {self._format_number(lambda_value)}\\)\n\n"
            "## Formula Used\n\n"
            "$$ P(X = x) = \\frac{e^{-\\lambda} \\lambda^x}{x!} $$\n\n"
            "## Step-by-step Solution\n\n"
            "Substitute the given values into the Poisson formula:\n\n"
            f"$$ P(X = {x_value}) = \\frac{{e^{{-{self._format_number(lambda_value)}}} ({self._format_number(lambda_value)})^{x_value}}}{{{x_value}!}} $$\n\n"
            f"$$ P(X = {x_value}) = {self._format_number(probability)} $$\n\n"
            "## Final Answer\n\n"
            f"**The required Poisson probability is \\({self._format_number(probability)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_binomial_mle(self, question, follow_up_prompt):
        lowered = question.lower()
        if "mle" not in lowered and "maximum likelihood" not in lowered:
            return None

        success_match = re.search(r"(\d+)\s+success", lowered)
        trial_match = re.search(r"(\d+)\s+(?:trial|observation|sample)", lowered)

        if not success_match or not trial_match:
            return None

        successes = int(success_match.group(1))
        trials = int(trial_match.group(1))
        if trials <= 0 or successes > trials:
            return None

        estimate = successes / trials
        response = (
            "## Given\n\n"
            f"- Number of successes: \\(x = {successes}\\)\n"
            f"- Number of trials: \\(n = {trials}\\)\n\n"
            "## Formula Used\n\n"
            "For a Bernoulli/Binomial model, the maximum likelihood estimator of \\(p\\) is:\n\n"
            "$$ \\hat{p} = \\frac{x}{n} $$\n\n"
            "## Step-by-step Solution\n\n"
            "Substitute the observed values:\n\n"
            f"$$ \\hat{{p}} = \\frac{{{successes}}}{{{trials}}} $$\n\n"
            f"$$ \\hat{{p}} = {self._format_number(estimate)} $$\n\n"
            "## Final Answer\n\n"
            f"**The maximum likelihood estimate is \\(\\hat{{p}} = {self._format_number(estimate)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_standard_normal_probability(self, question, follow_up_prompt):
        lowered = question.lower()
        normal_markers = (
            "standard normal",
            "normal distribution",
            "z table",
            "z-table",
            "p(z",
            "probability",
        )
        if not any(marker in lowered for marker in normal_markers):
            return None

        interval_bounds = self._extract_interval_bounds(question)
        if interval_bounds is not None and ("z" in lowered or "normal" in lowered):
            lower, upper = interval_bounds
            if lower > upper:
                lower, upper = upper, lower
            cdf_upper = self.standard_normal.cdf(upper)
            cdf_lower = self.standard_normal.cdf(lower)
            probability = cdf_upper - cdf_lower
            response = (
                "## Given\n\n"
                f"- Standard normal variable: \\(Z \\sim N(0,1)\\)\n"
                f"- Lower bound: \\({self._format_number(lower)}\\)\n"
                f"- Upper bound: \\({self._format_number(upper)}\\)\n\n"
                "## Formula Used\n\n"
                "$$ P(a < Z < b) = \\Phi(b) - \\Phi(a) $$\n\n"
                "## Step-by-step Solution\n\n"
                f"From the standard normal distribution:\n\n"
                f"$$ \\Phi({self._format_number(upper)}) = {self._format_number(cdf_upper)} $$\n\n"
                f"$$ \\Phi({self._format_number(lower)}) = {self._format_number(cdf_lower)} $$\n\n"
                "Now subtract the two cumulative probabilities:\n\n"
                f"$$ P({self._format_number(lower)} < Z < {self._format_number(upper)}) = {self._format_number(cdf_upper)} - {self._format_number(cdf_lower)} = {self._format_number(probability)} $$\n\n"
                "## Final Answer\n\n"
                f"**The required probability is \\({self._format_number(probability)}\\).**"
            )
            return self._append_follow_up(response, follow_up_prompt)

        z_match = re.search(r"p\s*\(\s*z\s*([<>=]+)\s*([-+]?\d*\.?\d+)\s*\)", question, re.IGNORECASE)
        if not z_match:
            z_match = re.search(r"z\s*([<>=]+)\s*([-+]?\d*\.?\d+)", question, re.IGNORECASE)
        if not z_match:
            return None

        operator = z_match.group(1)
        z_value = float(z_match.group(2))
        cdf_value = self.standard_normal.cdf(z_value)

        if operator in ("<", "<="):
            probability = cdf_value
            expression = f"P(Z \\le {self._format_number(z_value)})"
            explanation = (
                f"Using the standard normal table, the cumulative probability up to \\({self._format_number(z_value)}\\) is:\n\n"
                f"$$ \\Phi({self._format_number(z_value)}) = {self._format_number(cdf_value)} $$"
            )
        elif operator in (">", ">="):
            probability = 1 - cdf_value
            expression = f"P(Z > {self._format_number(z_value)})"
            explanation = (
                f"First find the left-tail probability:\n\n"
                f"$$ \\Phi({self._format_number(z_value)}) = {self._format_number(cdf_value)} $$\n\n"
                "Now use the complement rule:\n\n"
                f"$$ P(Z > {self._format_number(z_value)}) = 1 - {self._format_number(cdf_value)} = {self._format_number(probability)} $$"
            )
        elif operator == "=":
            probability = 0.0
            expression = f"P(Z = {self._format_number(z_value)})"
            explanation = (
                "For a continuous distribution, the probability at a single point is zero:\n\n"
                f"$$ P(Z = {self._format_number(z_value)}) = 0 $$"
            )
        else:
            return None

        response = (
            "## Given\n\n"
            f"- Standard normal variable: \\(Z \\sim N(0,1)\\)\n"
            f"- Required expression: \\({expression}\\)\n\n"
            "## Formula Used\n\n"
            "$$ \\Phi(z) = P(Z \\le z) $$\n\n"
            "## Step-by-step Solution\n\n"
            f"{explanation}\n\n"
            "## Final Answer\n\n"
            f"**The required probability is \\({self._format_number(probability)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_standard_normal_critical_value(self, question, follow_up_prompt):
        lowered = question.lower()
        if "critical value" not in lowered and "find z" not in lowered and "z value" not in lowered:
            return None
        if "confidence" not in lowered and "alpha" not in lowered and "significance" not in lowered:
            return None

        alpha = self._extract_number(
            question,
            [
                rf"alpha\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"significance(?: level)?\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        confidence_level = self._extract_number(
            question,
            [
                rf"({self.NUMBER_PATTERN})\s*%",
                rf"confidence level\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )

        if alpha is None and confidence_level is None:
            return None

        if confidence_level is not None:
            if confidence_level > 1:
                confidence_level /= 100
            alpha = 1 - confidence_level
        elif alpha is not None and alpha > 1:
            alpha /= 100

        if alpha is None or not (0 < alpha < 1):
            return None

        two_tailed = "two tailed" in lowered or "two-tailed" in lowered or "two sided" in lowered or "two-sided" in lowered
        upper_tail = alpha / 2 if two_tailed else alpha
        cumulative = 1 - upper_tail
        critical_value = self.standard_normal.inv_cdf(cumulative)

        tail_text = "two-tailed" if two_tailed else "one-tailed"
        alpha_term = "\\alpha/2" if two_tailed else "\\alpha"
        response = (
            "## Given\n\n"
            f"- Significance level: \\(\\alpha = {self._format_number(alpha)}\\)\n"
            f"- Test type: \\({tail_text}\\)\n\n"
            "## Formula Used\n\n"
            "$$ z_{\\alpha} = \\Phi^{-1}(1-\\alpha) \\quad \\text{or} \\quad z_{\\alpha/2} = \\Phi^{-1}(1-\\alpha/2) $$\n\n"
            "## Step-by-step Solution\n\n"
            f"For a {tail_text} setting, use the cumulative probability:\n\n"
            f"$$ 1 - {alpha_term} = {self._format_number(cumulative)} $$\n\n"
            "Now take the inverse standard normal value:\n\n"
            f"$$ z = \\Phi^{{-1}}({self._format_number(cumulative)}) = {self._format_number(critical_value)} $$\n\n"
            "## Final Answer\n\n"
            f"**The critical z-value is \\({self._format_number(critical_value)}\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def _solve_confidence_interval_mean_known_sigma(self, question, follow_up_prompt):
        lowered = question.lower()
        if "confidence interval" not in lowered and "confidence limits" not in lowered:
            return None
        if "sigma" not in lowered and "standard deviation" not in lowered:
            return None

        mean_value = self._extract_number(
            question,
            [
                rf"sample mean\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"mean\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"\\bar\{{x\}}\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        sigma_value = self._extract_number(
            question,
            [
                rf"(?:known\s+)?sigma\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"population standard deviation\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"standard deviation\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        sample_size = self._extract_int(
            question,
            [
                rf"sample size\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
                rf"\bn\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )
        confidence_level = self._extract_number(
            question,
            [
                rf"({self.NUMBER_PATTERN})\s*%",
                rf"confidence level\s*(?:=|is)?\s*({self.NUMBER_PATTERN})",
            ],
        )

        if (
            mean_value is None
            or sigma_value is None
            or sample_size is None
            or sample_size <= 0
            or confidence_level is None
        ):
            return None

        if confidence_level > 1:
            confidence_level /= 100

        z_lookup = {
            0.8: 1.281552,
            0.85: 1.439531,
            0.9: 1.644854,
            0.95: 1.959964,
            0.98: 2.326348,
            0.99: 2.575829,
        }
        z_value = z_lookup.get(round(confidence_level, 2))
        if z_value is None:
            return None

        standard_error = sigma_value / math.sqrt(sample_size)
        margin = z_value * standard_error
        lower = mean_value - margin
        upper = mean_value + margin

        response = (
            "## Given\n\n"
            f"- Sample mean: \\(\\bar{{x}} = {self._format_number(mean_value)}\\)\n"
            f"- Known population standard deviation: \\(\\sigma = {self._format_number(sigma_value)}\\)\n"
            f"- Sample size: \\(n = {sample_size}\\)\n"
            f"- Confidence level: \\({self._format_number(confidence_level * 100)}\\%\\)\n\n"
            "## Formula Used\n\n"
            "$$ \\bar{x} \\pm z_{\\alpha/2} \\frac{\\sigma}{\\sqrt{n}} $$\n\n"
            "## Step-by-step Solution\n\n"
            f"For a \\({self._format_number(confidence_level * 100)}\\%\\) confidence interval, use \\(z_{{\\alpha/2}} = {self._format_number(z_value)}\\).\n\n"
            "Compute the standard error:\n\n"
            f"$$ \\frac{{\\sigma}}{{\\sqrt{{n}}}} = \\frac{{{self._format_number(sigma_value)}}}{{\\sqrt{{{sample_size}}}}} = {self._format_number(standard_error)} $$\n\n"
            "Now compute the margin of error:\n\n"
            f"$$ {self._format_number(z_value)} \\times {self._format_number(standard_error)} = {self._format_number(margin)} $$\n\n"
            "Therefore the confidence interval is:\n\n"
            f"$$ {self._format_number(mean_value)} \\pm {self._format_number(margin)} $$\n\n"
            f"$$ ({self._format_number(lower)}, {self._format_number(upper)}) $$\n\n"
            "## Final Answer\n\n"
            f"**The confidence interval is \\(({self._format_number(lower)}, {self._format_number(upper)})\\).**"
        )
        return self._append_follow_up(response, follow_up_prompt)

    def solve(self, question, follow_up_prompt):
        for solver in (
            self._solve_z_score,
            self._solve_standard_normal_probability,
            self._solve_standard_normal_critical_value,
            self._solve_sample_mean,
            self._solve_variance_or_std,
            self._solve_binomial_probability,
            self._solve_poisson_probability,
            self._solve_binomial_mle,
            self._solve_confidence_interval_mean_known_sigma,
        ):
            result = solver(question, follow_up_prompt)
            if result:
                return result
        return None
