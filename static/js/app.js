const chatWindow=document.getElementById("chatWindow");
const input=document.getElementById("input");
const menuItems=document.querySelectorAll(".menu-item[data-view]");
const views={
"ask-ai":document.getElementById("ask-ai-view"),
"concepts":document.getElementById("concepts-view"),
"practice":document.getElementById("practice-view"),
"notes":document.getElementById("notes-view"),
"visualizations":document.getElementById("visualizations-view")
};
const conceptSearch=document.getElementById("conceptSearch");
const conceptBrowser=document.getElementById("conceptBrowser");
const conceptDetail=document.getElementById("conceptDetail");
const conceptBack=document.getElementById("conceptBack");
const conceptDetailTitle=document.getElementById("conceptDetailTitle");
const conceptDetailSubtitle=document.getElementById("conceptDetailSubtitle");
const conceptCards=[...document.querySelectorAll(".concept-card")];
const conceptVideoSuggestions=document.getElementById("conceptVideoSuggestions");
const conceptVideoSuggestionsTitle=document.getElementById("conceptVideoSuggestionsTitle");
const conceptVideoGrid=document.getElementById("conceptVideoGrid");
const subtopicStrip=document.getElementById("subtopicStrip");
const conceptOutput=document.getElementById("conceptOutput");
const conceptOutputTitle=document.getElementById("conceptOutputTitle");
const conceptAnswer=document.getElementById("conceptAnswer");
const practiceSearch=document.getElementById("practiceSearch");
const practiceBrowser=document.getElementById("practiceBrowser");
const practiceDetail=document.getElementById("practiceDetail");
const practiceBack=document.getElementById("practiceBack");
const practiceCards=[...document.querySelectorAll(".practice-card")];
const practiceTitle=document.getElementById("practiceTitle");
const practiceSubtitle=document.getElementById("practiceSubtitle");
const practiceVideoTitle=document.getElementById("practiceVideoTitle");
const practiceVideoGrid=document.getElementById("practiceVideoGrid");
const practiceChatShell=document.querySelector(".practice-chat-shell");
const practicePromptStrip=document.getElementById("practicePromptStrip");
const practiceChatWindow=document.getElementById("practiceChatWindow");
const practiceInput=document.getElementById("practiceInput");
const practiceSend=document.getElementById("practiceSend");
const notesSearch=document.getElementById("notesSearch");
const notesBrowser=document.getElementById("notesBrowser");
const notesDetail=document.getElementById("notesDetail");
const notesBack=document.getElementById("notesBack");
const notesCards=[...document.querySelectorAll(".notes-card")];
const notesTitle=document.getElementById("notesTitle");
const notesSubtitle=document.getElementById("notesSubtitle");
const notesSubtopicStrip=document.getElementById("notesSubtopicStrip");
const notesDownloadStatus=document.getElementById("notesDownloadStatus");
const notesDownloadBtn=document.getElementById("notesDownloadBtn");
const notesRender=document.getElementById("notesRender");
const notesVideoTitle=document.getElementById("notesVideoTitle");
const notesVideoGrid=document.getElementById("notesVideoGrid");
const notesChatWindow=document.getElementById("notesChatWindow");
const notesInput=document.getElementById("notesInput");
const notesSend=document.getElementById("notesSend");
const vizSearch=document.getElementById("vizSearch");
const vizBrowser=document.getElementById("vizBrowser");
const vizDetail=document.getElementById("vizDetail");
const vizBack=document.getElementById("vizBack");
const vizCards=[...document.querySelectorAll(".viz-card")];
const vizDetailTitle=document.getElementById("vizDetailTitle");
const vizDetailSubtitle=document.getElementById("vizDetailSubtitle");
const vizHero=document.getElementById("vizHero");
const vizSummary=document.getElementById("vizSummary");
const vizStatGrid=document.getElementById("vizStatGrid");
const vizInsights=document.getElementById("vizInsights");
const vizWhatItShows=document.getElementById("vizWhatItShows");
const vizWhenToUse=document.getElementById("vizWhenToUse");
const vizInterpret=document.getElementById("vizInterpret");
const vizMistakes=document.getElementById("vizMistakes");
const vizForm=document.getElementById("vizForm");
let currentPracticeExam="";
let currentNotesSection="";
let currentNotesSubtopic="";
const practicePromptMap={
"Descriptive Statistics and Probability":["Give me one practice question on descriptive statistics.","Ask me a Bayes theorem question.","Give me a hint only.","Now show the full solution."],
"Univariate Distributions":["Give me one univariate distributions question.","Ask me a Poisson or normal question.","Give me a hint only.","Now solve step by step."],
"Multivariate Distributions":["Give me one multivariate distributions question.","Ask me a joint and marginal distribution question.","Give me only the first hint.","Now give the full derivation."],
"Limit Theorems":["Give me one limit theorems question.","Ask me a CLT question.","Give a concept-first hint.","Now explain the full solution."],
"Sampling Distributions":["Give me one sampling distributions question.","Ask me a chi-square or t distribution question.","Give me a hint only.","Now show the full solution."],
"Estimation":["Give me one estimation question.","Ask me an MLE or UMVUE question.","Give me a short hint.","Now solve it completely."],
"Testing of Hypotheses":["Give me one testing of hypotheses question.","Ask me a Neyman-Pearson question.","Give me a hint only.","Now show the full solution."],
"Nonparametric Methods":["Give me one nonparametric methods question.","Ask me a Mann-Whitney test question.","Give me only the first hint.","Now give the full solution."],
"Stochastic Processes":["Give me one stochastic processes question.","Ask me a Markov chain question.","Give me a short hint.","Now solve step by step."]
};
const notesSubtopicMap={
"Descriptive Statistics and Probability":["Sample and Population","Types of Data","Tabular and Graphical Representation","Measures of Central Tendency","Measures of Dispersion","Moments, Skewness, Kurtosis","Covariance and Correlation","Axiomatic Probability","Conditional Probability","Bayes Theorem","Independence of Events"],
"Univariate Distributions":["Random Variables","CDF, PMF, PDF","Transformations and Jacobian","Mathematical Expectation","MGF and Uniqueness","Markov and Chebyshev Inequalities","Bernoulli and Binomial","Poisson and Geometric","Gamma and Beta","Normal and Cauchy"],
"Multivariate Distributions":["Random Vectors","Joint and Marginal Distributions","Conditional Distributions","Transformations of Random Vectors","Joint Moments","Covariance and Correlation","Conditional Expectation and Variance","Multinomial Distribution","Bivariate Normal Distribution"],
"Limit Theorems":["Modes of Convergence","Weak Law of Large Numbers","Strong Law of Large Numbers","Central Limit Theorem"],
"Sampling Distributions":["Sampling Distribution of a Statistic","Order Statistics","Smallest and Largest Order Statistics","Central Chi-square Distribution","Central t Distribution","Central F Distribution","Relationship between t, F and Chi-square"],
"Estimation":["Sufficiency and Factorization Theorem","Complete Statistic","Consistency and Efficiency","UMVUE","Rao-Blackwell and Lehmann-Scheffe","Cramer-Rao Inequality","Method of Moments","Maximum Likelihood Estimation","Least Squares Estimation","Confidence Intervals"],
"Testing of Hypotheses":["Null and Alternative Hypotheses","Type I and Type II Errors","Critical Region and Significance Level","Power of a Test","p-value","Most Powerful and UMP Tests","Neyman-Pearson Lemma","Likelihood Ratio Tests"],
"Nonparametric Methods":["Runs Test","Empirical Distribution Function","Kolmogorov-Smirnov Test","Sign Tests","Mann-Whitney Test"],
"Stochastic Processes":["Transition Probability Matrix","Higher Order Transition Probabilities","Markov Chain Graph","Chapman-Kolmogorov Equation","Classification of States and Chains","Stability of Markov Chain","Poisson Process","Interarrival and Waiting Times"]
};
const subtopicMap={
"Descriptive Statistics":[
{label:"Mean Median Mode", prompt:"Explain mean, median, and mode with intuition, formulas, and examples."},
{label:"Variance and SD", prompt:"Explain variance and standard deviation in descriptive statistics with formulas and practical meaning."},
{label:"Skewness", prompt:"Explain skewness in descriptive statistics, including positive and negative skew with examples."},
{label:"Kurtosis", prompt:"Explain kurtosis simply and how it describes tails and peakedness in data."},
{label:"Quartiles", prompt:"Explain quartiles, interquartile range, and boxplot interpretation with examples."},
{label:"Summary Measures", prompt:"Explain how descriptive statistics summarize a dataset using center, spread, and shape."}
],
"Inferential Statistics":[
{label:"Population vs Sample", prompt:"Explain the difference between population and sample in inferential statistics with examples."},
{label:"Sampling Error", prompt:"Explain sampling error and why it matters in inferential statistics."},
{label:"Confidence Intervals", prompt:"Explain confidence intervals with intuition and one simple example."},
{label:"Standard Error", prompt:"Explain standard error in inferential statistics with formula intuition and examples."},
{label:"Estimation", prompt:"Explain estimation in inferential statistics, including point and interval estimation."},
{label:"Inference Basics", prompt:"Explain inferential statistics from basics with simple real-world examples."}
],
"Probability Distributions":[
{label:"Random Variables", prompt:"Explain random variables in probability distributions, including discrete vs continuous random variables with examples."},
{label:"PMF and PDF", prompt:"Explain PMF and PDF clearly, including the difference between them and simple examples."},
{label:"CDF", prompt:"Explain cumulative distribution function (CDF) with intuition, graph meaning, and examples."},
{label:"Binomial", prompt:"Explain the binomial distribution with assumptions, formula, intuition, and one solved example."},
{label:"Poisson", prompt:"Explain the Poisson distribution with intuition, formula, use-cases, and one example."},
{label:"Normal", prompt:"Explain the normal distribution with mean, standard deviation, bell curve intuition, and examples."},
{label:"Exponential", prompt:"Explain the exponential distribution with formula, memoryless property, and practical examples."},
{label:"Expectation", prompt:"Explain expectation in probability distributions with formula, intuition, and examples."},
{label:"Variance", prompt:"Explain variance in probability distributions with formula, intuition, and examples."},
{label:"Compare Models", prompt:"Compare binomial, Poisson, normal, and exponential distributions with when to use each one."}
],
"Probability Theory":[
{label:"Sample Space", prompt:"Explain sample space and events in probability theory with examples."},
{label:"Conditional Probability", prompt:"Explain conditional probability with formula, intuition, and examples."},
{label:"Independence", prompt:"Explain independence of events in probability theory with examples."},
{label:"Bayes Theorem", prompt:"Explain Bayes theorem with intuition and one real-world example."},
{label:"Axioms", prompt:"Explain the basic axioms of probability in simple language."},
{label:"Set Operations", prompt:"Explain union, intersection, and complement in probability with examples."}
],
"Sampling Methods":[
{label:"Simple Random", prompt:"Explain simple random sampling with examples and use-cases."},
{label:"Stratified", prompt:"Explain stratified sampling with intuition, steps, and when to use it."},
{label:"Cluster", prompt:"Explain cluster sampling and how it differs from stratified sampling."},
{label:"Systematic", prompt:"Explain systematic sampling with one practical example."},
{label:"Bias", prompt:"Explain sampling bias and how poor sampling affects conclusions."},
{label:"Comparison", prompt:"Compare major sampling methods with advantages and disadvantages."}
],
"Regression Analysis":[
{label:"Linear Regression", prompt:"Explain linear regression with formula, intuition, and one solved example."},
{label:"Least Squares", prompt:"Explain the least squares idea in regression analysis simply."},
{label:"Residuals", prompt:"Explain residuals in regression and what they tell us."},
{label:"Assumptions", prompt:"Explain the assumptions of linear regression in simple terms."},
{label:"Multiple Regression", prompt:"Explain multiple regression and how it extends simple linear regression."},
{label:"Interpretation", prompt:"Explain how to interpret slope, intercept, and fitted line in regression analysis."}
],
"Hypothesis Testing":[
{label:"Null and Alternative", prompt:"Explain null and alternative hypotheses with examples."},
{label:"p-value", prompt:"Explain p-value in hypothesis testing with intuition and examples."},
{label:"Type I and II", prompt:"Explain Type I and Type II errors clearly with examples."},
{label:"z and t Tests", prompt:"Explain z-test and t-test, their differences, and when to use each."},
{label:"Chi-square", prompt:"Explain chi-square test with intuition and simple examples."},
{label:"Decision Rule", prompt:"Explain the decision rule in hypothesis testing using significance level and p-value."}
],
"Estimation Theory":[
{label:"Point Estimation", prompt:"Explain point estimation with simple examples."},
{label:"Interval Estimation", prompt:"Explain interval estimation and how it differs from point estimation."},
{label:"MLE", prompt:"Explain maximum likelihood estimation with intuition and one example."},
{label:"Method of Moments", prompt:"Explain method of moments estimation simply."},
{label:"Bias and Unbiasedness", prompt:"Explain bias and unbiasedness of estimators with examples."},
{label:"Estimator Properties", prompt:"Explain important estimator properties like consistency and efficiency."}
],
"Data Visualization":[
{label:"Histograms", prompt:"Explain histograms and what they reveal about data distribution."},
{label:"Scatter Plots", prompt:"Explain scatter plots and how to interpret relationships using them."},
{label:"Box Plots", prompt:"Explain box plots with quartiles, outliers, and interpretation."},
{label:"Line Charts", prompt:"Explain line charts and when they are useful in statistics."},
{label:"Bar Charts", prompt:"Explain bar charts and how they differ from histograms."},
{label:"Good Design", prompt:"Explain what makes a statistical visualization clear and effective."}
],
"ANOVA":[
{label:"Intuition", prompt:"Explain the intuition behind ANOVA in simple language."},
{label:"F Statistic", prompt:"Explain the F-statistic in ANOVA and how it is interpreted."},
{label:"Assumptions", prompt:"Explain the assumptions of ANOVA with examples."},
{label:"Between vs Within", prompt:"Explain between-group and within-group variance in ANOVA."},
{label:"One-way ANOVA", prompt:"Explain one-way ANOVA with one simple solved example."},
{label:"Post Hoc", prompt:"Explain why post hoc tests are needed after ANOVA."}
],
"Correlation and Association":[
{label:"Covariance", prompt:"Explain covariance and how it differs from correlation."},
{label:"Pearson", prompt:"Explain Pearson correlation with examples and interpretation."},
{label:"Spearman", prompt:"Explain Spearman rank correlation and when to use it."},
{label:"Positive vs Negative", prompt:"Explain positive, negative, and zero correlation with examples."},
{label:"Scatter Interpretation", prompt:"Explain how scatter plots help interpret association between variables."},
{label:"Limitations", prompt:"Explain why correlation does not imply causation."}
],
"Time Series Analysis":[
{label:"Trend", prompt:"Explain trend in time series analysis with examples."},
{label:"Seasonality", prompt:"Explain seasonality in time series with examples."},
{label:"Autocorrelation", prompt:"Explain autocorrelation in time series simply."},
{label:"Smoothing", prompt:"Explain smoothing methods in time series forecasting."},
{label:"Forecasting", prompt:"Explain forecasting basics in time series analysis."},
{label:"Components", prompt:"Explain the components of a time series: trend, seasonality, cyclical, and irregular."}
],
"Bayesian Statistics":[
{label:"Prior", prompt:"Explain prior distribution in Bayesian statistics with examples."},
{label:"Likelihood", prompt:"Explain likelihood in Bayesian statistics simply."},
{label:"Posterior", prompt:"Explain posterior distribution and how it is obtained."},
{label:"Bayes Update", prompt:"Explain Bayesian updating with one intuitive example."},
{label:"Frequentist vs Bayesian", prompt:"Compare frequentist and Bayesian statistics simply."},
{label:"Credible Intervals", prompt:"Explain credible intervals in Bayesian statistics."}
],
"Non-Parametric Statistics":[
{label:"Why Non-parametric", prompt:"Explain why non-parametric methods are used and when they are needed."},
{label:"Sign Test", prompt:"Explain the sign test with a simple example."},
{label:"Wilcoxon", prompt:"Explain the Wilcoxon test and when to use it."},
{label:"Mann-Whitney", prompt:"Explain the Mann-Whitney test in simple terms."},
{label:"Rank-based Methods", prompt:"Explain rank-based methods in non-parametric statistics."},
{label:"Comparison", prompt:"Compare parametric and non-parametric statistics with examples."}
],
"Multivariate Statistics":[
{label:"Multiple Variables", prompt:"Explain what multivariate statistics studies and why it matters."},
{label:"Covariance Matrix", prompt:"Explain covariance matrix with interpretation and examples."},
{label:"PCA", prompt:"Explain principal component analysis simply with intuition."},
{label:"Dimensionality Reduction", prompt:"Explain dimensionality reduction in multivariate statistics."},
{label:"Joint Distribution", prompt:"Explain joint behavior of multiple variables in multivariate analysis."},
{label:"Applications", prompt:"Explain practical applications of multivariate statistics."}
],
"Stochastic Processes":[
{label:"Random Process", prompt:"Explain what a stochastic process is with simple examples."},
{label:"Markov Chains", prompt:"Explain Markov chains with intuition and one example."},
{label:"Random Walk", prompt:"Explain random walk as a stochastic process with intuition."},
{label:"Poisson Process", prompt:"Explain Poisson process with practical examples."},
{label:"States and Transitions", prompt:"Explain states and transitions in stochastic processes."},
{label:"Time Evolution", prompt:"Explain how stochastic processes model systems evolving over time."}
]
};

const vizInfo={
"Histogram":{shape:"shape-bars",subtitle:"Interpret distribution shape, spread, and clustering.",what:"A histogram groups numeric values into bins and shows how frequently values fall in each interval. It helps you see the overall shape of a distribution.",when:"Use it when you want to inspect one quantitative variable and understand whether it looks symmetric, skewed, wide, narrow, or multi-modal.",interpret:["Look for peaks to identify common value regions.","Check skewness by seeing whether one tail extends farther.","Look for gaps or multiple peaks that may suggest subgroups."],mistakes:["Using too few bins can hide structure.","Using too many bins can create noisy patterns.","Treating it like a bar chart for categories is incorrect."],fields:[{name:"values",label:"Values (comma-separated)",value:"2,4,4,5,5,5,6,7,8,9,10,10,11,12"},{name:"bins",label:"Bins",value:"6"}]},
"Box Plot":{shape:"shape-box",subtitle:"Read quartiles, spread, and outliers quickly.",what:"A box plot summarizes a distribution using median, quartiles, whiskers, and outliers. It compresses key summary statistics into one compact graphic.",when:"Use it when comparing groups quickly or when you care about spread, central tendency, and outliers rather than exact frequencies.",interpret:["The line inside the box is the median.","The box spans the interquartile range.","Points beyond whiskers may indicate outliers."],mistakes:["Assuming it shows exact frequencies.","Ignoring sample size differences between groups.","Confusing whiskers with absolute minimum and maximum in every implementation."],fields:[{name:"values",label:"Values (comma-separated)",value:"4,5,5,6,6,7,8,9,10,12,15"}]},
"Scatter Plot":{shape:"shape-scatter",subtitle:"See relationships, trend, and clustering between variables.",what:"A scatter plot places one variable on each axis and shows paired observations as points. It reveals relationship patterns between two numeric variables.",when:"Use it for correlation, association, clustering, nonlinearity, and regression-related interpretation.",interpret:["Upward patterns suggest positive association.","Downward patterns suggest negative association.","Wide spread around a trend suggests weaker relationship."],mistakes:["Assuming correlation means causation.","Ignoring outliers that distort the pattern.","Using scatter plots for categorical-only variables."],fields:[{name:"x_values",label:"X values (comma-separated)",value:"1,2,3,4,5,6"},{name:"y_values",label:"Y values (comma-separated)",value:"2,3,5,4,6,7"}]},
"Normal Curve":{shape:"shape-curve",subtitle:"Understand symmetry, spread, and bell-shaped behavior.",what:"The normal curve is a smooth bell-shaped distribution centered around the mean. It is symmetric and defined by mean and standard deviation.",when:"Use it when teaching Gaussian behavior, z-scores, CLT intuition, and approximate probability reasoning.",interpret:["The center is the mean.","Width reflects standard deviation.","Most observations cluster near the center."],mistakes:["Assuming all data are normally distributed.","Forgetting that real datasets may only approximate normality.","Ignoring skewness or heavy tails."],fields:[{name:"mean",label:"Mean",value:"0"},{name:"std",label:"Standard deviation",value:"1"}]},
"Bar Chart":{shape:"shape-bars",subtitle:"Compare category values quickly and clearly.",what:"A bar chart compares magnitudes across categories using rectangular bars. It works well when the x-axis contains names or groups rather than continuous values.",when:"Use it for category comparison, counts, frequencies, or summary values across groups.",interpret:["Longer bars indicate larger values.","Large gaps highlight strong differences.","Category order affects readability."],mistakes:["Using too many categories.","Using it for continuous distribution shape instead of a histogram.","Overdecorating with unnecessary colors."],fields:[{name:"labels",label:"Labels (comma-separated)",value:"A,B,C,D"},{name:"values",label:"Values (comma-separated)",value:"12,19,7,14"}]},
"Pie Chart":{shape:"shape-pie",subtitle:"Show part-to-whole composition using slices.",what:"A pie chart represents how a whole is split among categories. Each slice size corresponds to a category's proportion of the total.",when:"Use it when you want to communicate simple composition with a small number of categories.",interpret:["Bigger slices indicate larger proportions.","The full circle represents 100 percent.","Small differences are harder to compare than in a bar chart."],mistakes:["Using too many slices.","Comparing categories with very similar proportions.","Using it when precise value comparison matters."],fields:[{name:"labels",label:"Labels (comma-separated)",value:"A,B,C,D"},{name:"values",label:"Values (comma-separated)",value:"30,20,25,25"}]},
"Pareto Distribution":{shape:"shape-pareto",subtitle:"Understand heavy-tail concentration and 80/20 style behavior.",what:"A Pareto distribution models situations where a small fraction of items contributes a large fraction of the total effect. It is a common heavy-tail model.",when:"Use it for wealth concentration, file sizes, insurance claims, reliability, and 80/20 style phenomena where extremes matter.",interpret:["Large rare values create a long right tail.","Small changes in the tail can affect totals strongly.","The distribution highlights concentration rather than symmetry."],mistakes:["Calling every skewed dataset Pareto.","Ignoring whether the tail actually follows a power-law pattern.","Using it without checking if extreme values dominate the data."],fields:[{name:"alpha",label:"Shape alpha",value:"2.5"},{name:"xm",label:"Minimum scale xm",value:"1"}]}
};

function renderVizForm(name, info){
if(!info.fields || !info.fields.length){
vizForm.innerHTML="<p style='color:var(--muted)'>Interactive controls for this visualization will be added next.</p>";
vizHero.className=`viz-hero ${info.shape}`;
vizHero.innerHTML="";
vizSummary.classList.remove("active");
return;
}

vizForm.innerHTML=info.fields.map(field=>`
<div class="viz-field">
<label for="viz-${field.name}">${field.label}</label>
<input id="viz-${field.name}" name="${field.name}" value="${field.value}">
</div>`).join("") + `<button type="submit" class="viz-generate">Generate Graph</button>`;

vizForm.onsubmit=async e=>{
e.preventDefault();
const formData=new FormData(vizForm);
const payload={chart_type:name};
for(const [key,value] of formData.entries()){
payload[key]=value;
}
vizHero.className=`viz-hero ${info.shape}`;
vizHero.innerHTML="<div style='padding:24px;color:#e2e8f0;font-weight:700;'>Generating graph...</div>";
vizSummary.classList.remove("active");
try{
const res=await fetch("/visualize",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(payload)
});
const data=await res.json();
if(!res.ok){
throw new Error(data.error || "Visualization failed.");
}
vizHero.className="viz-hero has-output";
vizHero.innerHTML=`<img src="data:image/png;base64,${data.image}" alt="Generated visualization">`;
vizWhatItShows.textContent=data.explanation || info.what;
const stats=data.stats || {};
vizStatGrid.innerHTML=Object.entries(stats).map(([label,value])=>`
<div class="viz-stat">
<strong>${label}</strong>
<span>${value}</span>
</div>`).join("");
const insights=data.insights || [];
vizInsights.innerHTML=insights.map(item=>`<li>${item}</li>`).join("");
vizSummary.classList.toggle("active",Object.keys(stats).length > 0 || insights.length > 0);
vizHero.scrollIntoView({behavior:"smooth", block:"start"});
}catch(error){
vizHero.className=`viz-hero ${info.shape}`;
vizHero.innerHTML=`<div style='padding:24px;color:#fecaca;font-weight:700;'>Error: ${error.message}</div>`;
vizSummary.classList.remove("active");
}
};
}

menuItems.forEach(item=>{
item.addEventListener("click",()=>{
const target=item.dataset.view;
menuItems.forEach(node=>node.classList.remove("active"));
item.classList.add("active");
Object.values(views).forEach(view=>view.classList.remove("active"));
if(views[target]) views[target].classList.add("active");
});
});

conceptSearch.addEventListener("input",e=>{
const query=e.target.value.trim().toLowerCase();
conceptCards.forEach(card=>{
const title=card.querySelector("h3").innerText.toLowerCase();
const tags=(card.dataset.tags || "").toLowerCase();
const matches=!query || title.includes(query) || tags.includes(query);
card.classList.toggle("is-hidden",!matches);
});
});

practiceSearch.addEventListener("input",e=>{
const query=e.target.value.trim().toLowerCase();
practiceCards.forEach(card=>{
const title=card.querySelector("h3").innerText.toLowerCase();
const tags=(card.dataset.tags || "").toLowerCase();
const matches=!query || title.includes(query) || tags.includes(query);
card.classList.toggle("is-hidden",!matches);
});
});

notesSearch.addEventListener("input",e=>{
const query=e.target.value.trim().toLowerCase();
notesCards.forEach(card=>{
const title=card.querySelector("h3").innerText.toLowerCase();
const tags=(card.dataset.tags || "").toLowerCase();
const matches=!query || title.includes(query) || tags.includes(query);
card.classList.toggle("is-hidden",!matches);
});
});

conceptBack.addEventListener("click",()=>{
conceptDetail.classList.remove("active");
conceptBrowser.classList.add("active");
});

practiceBack.addEventListener("click",()=>{
practiceDetail.classList.remove("active");
practiceBrowser.classList.add("active");
currentPracticeExam="";
practiceChatWindow.innerHTML="";
practiceInput.value="";
practiceChatShell.style.display="grid";
});

notesBack.addEventListener("click",()=>{
notesDetail.classList.remove("active");
notesBrowser.classList.add("active");
currentNotesSection="";
currentNotesSubtopic="";
notesChatWindow.innerHTML="";
notesInput.value="";
notesSubtopicStrip.innerHTML="";
notesVideoGrid.innerHTML="";
notesRender.innerHTML="";
notesRender.classList.remove("active");
notesDownloadBtn.style.display="none";
notesDownloadBtn.dataset.url="";
notesDownloadStatus.textContent="Select a subtopic to generate the note preview and PDF.";
});

vizSearch.addEventListener("input",e=>{
const query=e.target.value.trim().toLowerCase();
vizCards.forEach(card=>{
const title=card.querySelector("h3").innerText.toLowerCase();
const tags=(card.dataset.tags || "").toLowerCase();
const matches=!query || title.includes(query) || tags.includes(query);
card.classList.toggle("is-hidden",!matches);
});
});

vizBack.addEventListener("click",()=>{
vizDetail.classList.remove("active");
vizBrowser.classList.add("active");
});

conceptCards.forEach(card=>{
const cta=card.querySelector(".concept-cta");
const handleConceptClick=()=>{
const concept=card.dataset.concept || card.querySelector("h3").innerText;
const action=card.dataset.action || "Start Analysis";
launchConceptPrompt(concept,action);
};
card.addEventListener("click",handleConceptClick);
if(cta){
cta.addEventListener("click",e=>{
e.stopPropagation();
handleConceptClick();
});
}
});

practiceCards.forEach(card=>{
const cta=card.querySelector(".concept-cta");
const open=()=>{
const exam=card.dataset.exam || card.querySelector("h3").innerText;
launchPractice(exam);
};
card.addEventListener("click",open);
if(cta){
cta.addEventListener("click",e=>{
e.stopPropagation();
open();
});

notesCards.forEach(card=>{
const cta=card.querySelector(".concept-cta");
const open=()=>{
const section=card.dataset.section || card.querySelector("h3").innerText;
launchNotes(section);
};
card.addEventListener("click",open);
if(cta){
cta.addEventListener("click",e=>{
e.stopPropagation();
open();
});
}
});
}
});

vizCards.forEach(card=>{
const open=()=>{
const key=card.dataset.viz;
const info=vizInfo[key];
if(!info) return;
vizBrowser.classList.remove("active");
vizDetail.classList.add("active");
vizDetailTitle.textContent=key;
vizDetailSubtitle.textContent=info.subtitle;
vizHero.className=`viz-hero ${info.shape}`;
vizHero.innerHTML="";
vizSummary.classList.remove("active");
vizStatGrid.innerHTML="";
vizInsights.innerHTML="";
vizWhatItShows.textContent=info.what;
vizWhenToUse.textContent=info.when;
vizInterpret.innerHTML=info.interpret.map(item=>`<li>${item}</li>`).join("");
vizMistakes.innerHTML=info.mistakes.map(item=>`<li>${item}</li>`).join("");
renderVizForm(key, info);
};
card.addEventListener("click",open);
const btn=card.querySelector(".viz-cta");
if(btn){
btn.addEventListener("click",e=>{
e.stopPropagation();
open();
});
}
});

function launchConceptPrompt(concept,action){
conceptBrowser.classList.remove("active");
conceptDetail.classList.add("active");
conceptDetailTitle.textContent=concept;
conceptDetailSubtitle.textContent="Explore recommended videos first, then pick a subtopic to get the explanation.";
loadConceptResources(concept,action);
}

function renderNotesVideos(videos){
if(Array.isArray(videos) && videos.length){
notesVideoGrid.innerHTML=videos.map(video=>`
<div class="video-card">
<iframe
class="video-frame"
src="${video.embed_url || ''}"
title="${video.title}"
loading="lazy"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
allowfullscreen></iframe>
<div>
<h5>${video.title}</h5>
<p>${video.description || "Topic-related video."}</p>
<div class="video-meta">${video.channel || "YouTube"}${video.published_at ? " • " + new Date(video.published_at).toLocaleDateString() : ""}</div>
</div>
</div>`).join("");
}else{
notesVideoGrid.innerHTML="<div class='video-meta'>No videos found for this subtopic right now.</div>";
}
}

async function loadNoteSubtopic(subtopic){
currentNotesSubtopic=subtopic;
notesDownloadStatus.textContent="Preparing note preview, PDF, and videos...";
notesVideoTitle.textContent=`${subtopic} Videos`;
notesVideoGrid.innerHTML="<div class='video-meta'>Loading videos...</div>";
notesChatWindow.innerHTML="";
notesRender.innerHTML="";
notesRender.classList.remove("active");
notesDownloadBtn.style.display="none";
notesDownloadBtn.dataset.url="";
appendNotes("bot",`Notes workspace ready for **${subtopic}** under **${currentNotesSection}**. Ask for explanation, examples, derivations, or exam tips.`);
try{
const [resourceRes, contentRes] = await Promise.all([
fetch("/notes-resource",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({section:currentNotesSection,subtopic})
}),
fetch("/notes-content",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({section:currentNotesSection,subtopic})
})
]);
const resourceData=await resourceRes.json();
const contentData=await contentRes.json();
if(!resourceRes.ok || !contentRes.ok){
 throw new Error((resourceData && resourceData.error) || (contentData && contentData.error) || "Notes generation failed");
}
renderNotesVideos(resourceData.videos || []);
notesRender.innerHTML=marked.parse(normalizeMathText(contentData.content || ""));
notesRender.classList.add("active");
if(window.MathJax){MathJax.typesetPromise();}
notesDownloadBtn.dataset.url=resourceData.download_url || "";
notesDownloadBtn.style.display=resourceData.download_url ? "inline-flex" : "none";
notesDownloadStatus.textContent=`Note ready for ${subtopic}. Preview loaded below and PDF download is available.`;
appendNotes("bot",`The note for **${subtopic}** is ready. Use **Download PDF** if you want the file, or ask me anything about this subtopic.`);
}catch(error){
notesVideoGrid.innerHTML="<div class='video-meta'>Videos could not be loaded right now.</div>";
notesDownloadStatus.textContent=`Notes could not be prepared right now.`;
notesRender.classList.remove("active");
appendNotes("bot",`Notes generation failed for **${subtopic}**. The backend likely needs a restart to load the new Notes routes.`);
}
}

function renderNotesSubtopics(section){
const items=notesSubtopicMap[section] || [];
notesSubtopicStrip.innerHTML=items.map(item=>`<button class="subtopic-chip" data-subtopic="${item.replace(/"/g,"&quot;")}">${item}</button>`).join("");
notesSubtopicStrip.querySelectorAll(".subtopic-chip").forEach(button=>{
button.addEventListener("click",()=>loadNoteSubtopic(button.dataset.subtopic));
});
}

function launchNotes(section){
currentNotesSection=section;
currentNotesSubtopic="";
notesBrowser.classList.remove("active");
notesDetail.classList.add("active");
notesTitle.textContent=section;
notesSubtitle.textContent="Pick a subtopic to download notes, watch videos, and ask follow-up questions.";
notesDownloadStatus.textContent="Select a subtopic to generate the note preview and PDF.";
notesVideoTitle.textContent="Related Videos";
notesVideoGrid.innerHTML="";
notesChatWindow.innerHTML="";
notesRender.innerHTML="";
notesRender.classList.remove("active");
notesDownloadBtn.style.display="none";
notesDownloadBtn.dataset.url="";
notesInput.value="";
renderNotesSubtopics(section);
}

function renderPracticePrompts(exam){
const prompts=practicePromptMap[exam] || [];
practicePromptStrip.innerHTML=prompts.map(prompt=>`<button class="practice-prompt" data-prompt="${prompt.replace(/"/g,"&quot;")}">${prompt}</button>`).join("");
practicePromptStrip.querySelectorAll(".practice-prompt").forEach(button=>{
button.addEventListener("click",()=>{
practiceInput.value=button.dataset.prompt;
sendPractice();
});
});
}

function renderPracticeVideos(videos){
if(Array.isArray(videos) && videos.length){
practiceVideoGrid.innerHTML=videos.map(video=>`
<div class="video-card">
<iframe
class="video-frame"
src="${video.embed_url || ''}"
title="${video.title}"
loading="lazy"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
allowfullscreen></iframe>
<div>
<h5>${video.title}</h5>
<p>${video.description || "Embedded practice video."}</p>
<div class="video-meta">${video.channel || "YouTube"}${video.published_at ? " â€¢ " + new Date(video.published_at).toLocaleDateString() : ""}</div>
</div>
</div>`).join("");
}else{
practiceVideoGrid.innerHTML="<div class='video-meta'>No videos found right now.</div>";
}
}

async function loadPracticeResources(exam){
practiceVideoTitle.textContent="Related Videos";
practiceVideoGrid.innerHTML="<div class='video-meta'>Loading Dr. Harish Garg practice videos...</div>";
practiceChatWindow.innerHTML="";
practiceChatShell.style.display="grid";
appendPractice("bot",`Practice workspace ready for **${exam}**. Ask for a question, a hint, a topic-wise problem, or a full solution.`);
renderPracticePrompts(exam);
try{
const res=await fetch("/practice-analysis",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({exam})
});
const data=await res.json();
if(!res.ok){
practiceVideoGrid.innerHTML="<div class='video-meta'>Practice videos could not be loaded.</div>";
return;
}
practiceSubtitle.textContent=data.subtitle || "Topic practice videos and chatbot support.";
renderPracticeVideos(data.videos || []);
if(data.starter){
appendPractice("bot",`Starter prompt: ${data.starter}`);
}
}catch(error){
practiceVideoGrid.innerHTML="<div class='video-meta'>Practice videos could not be loaded.</div>";
}
}

function launchPractice(exam){
currentPracticeExam=exam;
practiceBrowser.classList.remove("active");
practiceDetail.classList.add("active");
practiceTitle.textContent=exam;
practiceSubtitle.textContent="Loading practice videos and workspace...";
loadPracticeResources(exam);
}

function normalizeMathText(text){
return String(text || "")
.replace(/\\\\\(/g, "$")
.replace(/\\\\\)/g, "$")
.replace(/\\\\\[/g, "$$")
.replace(/\\\\\]/g, "$$")
.replace(/\\\$/g, "$")
.replace(/\u2212/g, "-");
}

function formatStructuredAnswer(text){
let formatted=normalizeMathText(text)
.replace(/^Definition:\s*/im,"**Definition:** ")
.replace(/^Mathematical Form:\s*/im,"\n\n**Mathematical Form:**\n")
.replace(/^Key Points:\s*/im,"\n\n**Key Points:**\n")
.replace(/^Source pages:\s*/im,"\n\n**Source pages:** ");

formatted=formatted.replace(
/(\*\*Mathematical Form:\*\*\n)([\s\S]*?)(\n\n\*\*Key Points:\*\*|\n\n\*\*Source pages:\*\*|$)/i,
(_,label,body,tail)=>{
const formulaBody=body.trim();
if(!formulaBody){return `${label}Not applicable${tail}`;}
if(/not applicable/i.test(formulaBody)){return `${label}Not applicable${tail}`;}
if(/\$\$[\s\S]*\$\$|\$[^$]+\$|\\\[[\s\S]*\\\]|\\\([\s\S]*\\\)/.test(formulaBody)){
return `${label}${formulaBody}${tail}`;
}
const cleaned=formulaBody
.split("\n")
.map(line=>line.trim())
.filter(Boolean)
.join(" ");
return `${label}$$${cleaned}$$${tail}`;
}
);

return formatted;
}

function append(role,text){
const msg=document.createElement("div");
msg.className="msg "+role;
msg.innerHTML=marked.parse(formatStructuredAnswer(text));
chatWindow.appendChild(msg);
chatWindow.scrollTop=chatWindow.scrollHeight;
if(window.MathJax){MathJax.typesetPromise();}
}

function appendPractice(role,text){
const msg=document.createElement("div");
msg.className="msg "+role;
msg.innerHTML=marked.parse(formatStructuredAnswer(text));
practiceChatWindow.appendChild(msg);
practiceChatWindow.scrollTop=practiceChatWindow.scrollHeight;
if(window.MathJax){MathJax.typesetPromise();}
}

function appendNotes(role,text){
const msg=document.createElement("div");
msg.className="msg "+role;
msg.innerHTML=marked.parse(formatStructuredAnswer(text));
notesChatWindow.appendChild(msg);
notesChatWindow.scrollTop=notesChatWindow.scrollHeight;
if(window.MathJax){MathJax.typesetPromise();}
}

function renderVideos(videos,title){
if(Array.isArray(videos) && videos.length){
conceptVideoSuggestionsTitle.textContent=title || "Related Videos";
conceptVideoGrid.innerHTML=videos.map(video=>`
<div class="video-card">
<iframe
class="video-frame"
src="${video.embed_url || ''}"
title="${video.title}"
loading="lazy"
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
allowfullscreen></iframe>
<div>
<h5>${video.title}</h5>
<p>${video.description || "Open this YouTube video for a topic-focused explanation."}</p>
<div class="video-meta">${video.channel || "YouTube"}${video.published_at ? " • " + new Date(video.published_at).toLocaleDateString() : ""}</div>
</div>
</div>`).join("");
conceptVideoSuggestions.classList.add("active");
}else{
conceptVideoGrid.innerHTML="";
conceptVideoSuggestions.classList.remove("active");
}
}

function renderSubtopics(concept){
const items=subtopicMap[concept] || [];
if(!items.length){
subtopicStrip.innerHTML="";
subtopicStrip.classList.remove("active");
return;
}

subtopicStrip.innerHTML=items.map(item=>`<button class="subtopic-chip" data-prompt="${item.prompt.replace(/"/g,"&quot;")}">${item.label}</button>`).join("");
subtopicStrip.classList.add("active");

subtopicStrip.querySelectorAll(".subtopic-chip").forEach(button=>{
button.addEventListener("click",async ()=>{
conceptOutputTitle.textContent=button.textContent;
conceptAnswer.innerHTML="<em>Generating explanation...</em>";
conceptOutput.classList.add("active");
try{
const res=await fetch("/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({message:button.dataset.prompt})
});
const data=await res.json();
conceptAnswer.innerHTML=marked.parse(normalizeMathText(data.reply || "No explanation available."));
if(window.MathJax){MathJax.typesetPromise();}
}catch(error){
conceptAnswer.innerHTML="<strong>Error:</strong> Concept explanation could not be loaded.";
}
});
});
}

async function fetchConceptVideos(concept,action){
try{
conceptVideoSuggestionsTitle.textContent="Related Videos";
conceptVideoGrid.innerHTML="<div class='video-meta'>Loading recommended NPTEL videos...</div>";
subtopicStrip.innerHTML="";
subtopicStrip.classList.remove("active");
conceptVideoSuggestions.classList.add("active");
const res=await fetch("/concept-analysis",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({concept,action,mode:"videos_only"})
});
const data=await res.json();
if(!res.ok){
conceptVideoSuggestions.classList.remove("active");
return null;
}
renderVideos(data.videos, `Related Videos`);
renderSubtopics(concept);
return data.videos || [];
}catch(error){
conceptVideoSuggestions.classList.remove("active");
return null;
}
}

async function loadConceptResources(concept,action){
conceptVideoSuggestionsTitle.textContent=concept;
conceptVideoGrid.innerHTML="";
subtopicStrip.innerHTML="";
subtopicStrip.classList.remove("active");
conceptVideoSuggestions.classList.add("active");
conceptOutput.classList.remove("active");
conceptAnswer.innerHTML="";
conceptOutputTitle.textContent="Concept Explanation";
await fetchConceptVideos(concept,action);
}

async function send(context={}){
const text=input.value.trim();
if(!text)return;

append("user",text);
input.value="";

const thinking=document.createElement("div");
thinking.className="msg bot";
thinking.innerHTML="<em>Analyzing statistical context...</em>";
chatWindow.appendChild(thinking);
chatWindow.scrollTop=chatWindow.scrollHeight;

if(context.concept){
await fetchConceptVideos(context.concept,context.action || "Start Analysis");
}

try{
const res=await fetch("/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({message:text})
});
const data=await res.json();
thinking.remove();

let i=0;
const full=data.reply;
const bot=document.createElement("div");
bot.className="msg bot";
chatWindow.appendChild(bot);

function type(){
bot.innerText=full.slice(0,i);
chatWindow.scrollTop=chatWindow.scrollHeight;
if(i<full.length){
i++;
setTimeout(type,15);
}else{
bot.innerHTML=marked.parse(normalizeMathText(full));
if(window.MathJax){MathJax.typesetPromise();}
}
}
type();

}catch(e){
thinking.remove();
append("bot","**Error:** Backend not reachable.");
}
}

async function sendPractice(){
const text=practiceInput.value.trim();
if(!text || !currentPracticeExam) return;

appendPractice("user",text);
practiceInput.value="";

const thinking=document.createElement("div");
thinking.className="msg bot";
thinking.innerHTML="<em>Preparing practice response...</em>";
practiceChatWindow.appendChild(thinking);
practiceChatWindow.scrollTop=practiceChatWindow.scrollHeight;

try{
const res=await fetch("/practice-chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({exam:currentPracticeExam,message:text})
});
const data=await res.json();
thinking.remove();
appendPractice("bot",data.reply || "No response available.");
}catch(error){
thinking.remove();
appendPractice("bot","**Error:** Practice chatbot is not reachable.");
}
}

async function sendNotes(){
const text=notesInput.value.trim();
if(!text || !currentNotesSection || !currentNotesSubtopic) return;

appendNotes("user",text);
notesInput.value="";

const thinking=document.createElement("div");
thinking.className="msg bot";
thinking.innerHTML="<em>Preparing notes response...</em>";
notesChatWindow.appendChild(thinking);
notesChatWindow.scrollTop=notesChatWindow.scrollHeight;

try{
const res=await fetch("/notes-chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({section:currentNotesSection,subtopic:currentNotesSubtopic,message:text})
});
const data=await res.json();
thinking.remove();
appendNotes("bot",data.reply || "No response available.");
}catch(error){
thinking.remove();
appendNotes("bot","**Error:** Notes chatbot is not reachable.");
}
}

input.addEventListener("keydown",e=>{
if(e.key==="Enter")send();
});

practiceInput.addEventListener("keydown",e=>{
if(e.key==="Enter")sendPractice();
});

practiceSend.addEventListener("click",sendPractice);
notesInput.addEventListener("keydown",e=>{
if(e.key==="Enter")sendNotes();
});
notesSend.addEventListener("click",sendNotes);
notesDownloadBtn.addEventListener("click",()=>{
const url=notesDownloadBtn.dataset.url;
if(!url) return;
const link=document.createElement("a");
link.href=url;
link.download=`${(currentNotesSubtopic || "notes").replace(/[^A-Za-z0-9]+/g,"_")}.pdf`;
document.body.appendChild(link);
link.click();
link.remove();
});
