(() => {
  const config = window.RETAINX_CONFIG || { models: {}, aiConfigured: false };
  const form = document.getElementById("customer-form");
  const industrySelect = document.getElementById("industry");
  const domainTabs = document.querySelectorAll(".domain-tab");
  const sections = [...document.querySelectorAll(".industry-fields")];
  const errorBox = document.getElementById("error-message");
  const analyzeButton = document.getElementById("analyze-button");
  const consent = document.getElementById("ai-consent");
  
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatSubmit = document.getElementById("chat-submit");
  const chatMessages = document.getElementById("chat-messages");
  const recommendationCard = document.getElementById("recommendation-card");
  const promptChips = document.querySelectorAll(".chip-btn");

  const history = [];
  let latestCustomer = null;
  let latestIndustry = "ecommerce";

  // Calibrated scenario presets for quick testing
  const presets = {
    ecommerce: {
      critical: {
        days_since_last_order: 120,
        orders_in_history: 1,
        days_since_first_order: 120,
        average_order_value: 1450,
        total_historical_spend: 1450,
        average_items_per_order: 1,
        last_order_value: 1450,
        last_freight_value: 380,
        last_payment_installments: 1,
        last_payment_type: "credit_card",
        average_review_score: 1,
        last_review_score: 1,
        average_delivery_delay_days: 14,
        average_delivery_duration_days: 28,
        review_text: "O produto atrasou mais de 14 dias além do prazo prometido e a caixa chegou amassada e rasgada. Experiência péssima com o vendedor."
      },
      borderline: {
        days_since_last_order: 75,
        orders_in_history: 2,
        days_since_first_order: 180,
        average_order_value: 850,
        total_historical_spend: 1700,
        average_items_per_order: 1,
        last_order_value: 850,
        last_freight_value: 240,
        last_payment_installments: 1,
        last_payment_type: "boleto",
        average_review_score: 3,
        last_review_score: 3,
        average_delivery_delay_days: 2,
        average_delivery_duration_days: 15,
        review_text: "O produto em si é aceitável, mas paguei muito caro no frete e demorou mais do que o esperado para chegar."
      },
      healthy: {
        days_since_last_order: 20,
        orders_in_history: 5,
        days_since_first_order: 360,
        average_order_value: 3200,
        total_historical_spend: 16000,
        average_items_per_order: 3,
        last_order_value: 3200,
        last_freight_value: 80,
        last_payment_installments: 6,
        last_payment_type: "credit_card",
        average_review_score: 5,
        last_review_score: 5,
        average_delivery_delay_days: -4,
        average_delivery_duration_days: 7,
        review_text: "Adorei a compra! O produto é de altíssima qualidade e chegou quatro dias antes da data estimada. Com certeza comprarei novamente."
      }
    }
  };

  const featureLabels = {
    days_since_last_order: "Days Since Last Order",
    average_delivery_delay_days: "Carrier SLA Delay (Days)",
    average_delivery_duration_days: "Total Transit Duration",
    average_items_per_order: "Basket Item Count",
    total_historical_spend: "Lifetime Spend (₹)",
    last_order_value: "Last Order Spend",
    last_freight_value: "Shipping Cost",
    last_payment_installments: "Installment Count",
    last_review_score: "Last Review Rating",
    orders_in_history: "Historical Orders Count",
    days_since_first_order: "Relationship Tenure (Days)",
    average_order_value: "Average Order Value (AOV)",
    average_review_score: "Average Review Stars",
    last_payment_type: "Payment Channel"
  };

  function displayFeature(name) {
    const category = name.match(/^([^=]+)=(.+)$/);
    if (category) {
      const label = featureLabels[category[1]] || category[1].replaceAll("_", " ");
      return `${label}: ${category[2].replaceAll("_", " ")}`;
    }
    return featureLabels[name] || name.replaceAll("_", " ");
  }

  // Executive-grade Markdown to HTML Renderer
  function renderMarkdown(md) {
    if (!md) return "";
    let html = md
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks
    html = html.replace(/```([\s\S]*?)```/gm, "<pre class=\"chat-code-block\"><code>$1</code></pre>");
    
    // Headings
    html = html.replace(/^#### (.*$)/gim, "<h5>$1</h5>");
    html = html.replace(/^### (.*$)/gim, "<h4>$1</h4>");
    html = html.replace(/^## (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^# (.*$)/gim, "<h2>$1</h2>");

    // Bold, Italic, Inline Code
    html = html.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/gim, "<em>$1</em>");
    html = html.replace(/`([^`]+)`/g, "<code class=\"inline-code\">$1</code>");

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, "<blockquote>$1</blockquote>");

    // Lists
    html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, "<li>$1</li>");
    html = html.replace(/^\s*\d+\.\s+(.*$)/gim, "<li class=\"num-li\">$1</li>");

    // Wrap list items
    html = html.replace(/(<li>.*?<\/li>)/gms, "<ul>$1</ul>");
    html = html.replace(/<\/ul>\s*<ul>/g, "");

    const blocks = html.split(/\n{2,}/).map(block => {
      block = block.trim();
      if (!block) return "";
      if (
        block.startsWith("<h") ||
        block.startsWith("<ul") ||
        block.startsWith("<ol") ||
        block.startsWith("<blockquote") ||
        block.startsWith("<pre")
      ) {
        return block;
      }
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    });

    return blocks.join("");
  }

  function setIndustry(industry) {
    latestIndustry = industry;
    if (industrySelect) industrySelect.value = industry;

    domainTabs.forEach(tab => {
      tab.classList.toggle("active", tab.dataset.domain === industry);
    });

    sections.forEach(section => {
      const active = section.dataset.industry === industry;
      section.classList.toggle("hidden", !active);
      section.querySelectorAll("input, select").forEach(input => {
        input.disabled = !active;
      });
    });

    const hintEl = document.getElementById("industry-hint");
    if (hintEl) {
      if (industry === "ecommerce") {
        hintEl.textContent = "Trained on 100,000+ customer records from Olist E-Commerce dataset.";
      } else {
        hintEl.textContent = "Enterprise retention intelligence.";
      }
    }

    if (analyzeButton) {
      analyzeButton.disabled = config.models ? !config.models[industry] : false;
    }
  }

  domainTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      setIndustry(tab.dataset.domain);
      clearError();
      resetResults();
    });
  });

  function loadPreset(profileType) {
    const currentIndustry = latestIndustry || "ecommerce";
    const data = presets[currentIndustry]?.[profileType];
    if (!data) return;

    const activeSection = sections.find(s => s.dataset.industry === currentIndustry);
    if (!activeSection) return;

    Object.entries(data).forEach(([key, val]) => {
      if (key === "review_text") {
        const reviewEl = document.getElementById("review-text");
        if (reviewEl) reviewEl.value = val;
        return;
      }
      const field = activeSection.querySelector(`[name="${key}"]`);
      if (field) {
        field.value = String(val);
      }
    });

    if (form) form.dispatchEvent(new Event("submit"));
  }

  const pCrit = document.getElementById("preset-critical");
  const pBord = document.getElementById("preset-borderline");
  const pHeal = document.getElementById("preset-healthy");
  if (pCrit) pCrit.addEventListener("click", () => loadPreset("critical"));
  if (pBord) pBord.addEventListener("click", () => loadPreset("borderline"));
  if (pHeal) pHeal.addEventListener("click", () => loadPreset("healthy"));

  function showError(message) {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function clearError() {
    if (!errorBox) return;
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function resetResults() {
    latestCustomer = null;
    const predRes = document.getElementById("prediction-result");
    const emptyRes = document.getElementById("empty-result");
    if (predRes) predRes.classList.add("hidden");
    if (emptyRes) emptyRes.classList.remove("hidden");
    if (recommendationCard) recommendationCard.classList.add("hidden");
    if (chatInput) chatInput.disabled = true;
    if (chatSubmit) chatSubmit.disabled = true;
    clearChat();
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const contentType = response.headers.get("content-type") || "";
    let data;
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      const text = await response.text();
      throw new Error(`Server returned HTTP ${response.status} (${response.statusText || "non-JSON"}).`);
    }
    if (!response.ok) {
      throw new Error(data.error || "The service request could not be completed.");
    }
    return data;
  }

  function collectCustomer() {
    const activeSection = sections.find(s => s.dataset.industry === latestIndustry);
    const customer = {};
    if (activeSection) {
      activeSection.querySelectorAll("input, select").forEach(input => {
        if (input.type === "number") {
          customer[input.name] = Number(input.value);
        } else if (input.name === "complaints") {
          customer[input.name] = input.value === "true";
        } else {
          customer[input.name] = input.value;
        }
      });
    }
    const reviewEl = document.getElementById("review-text");
    customer.review_text = reviewEl ? reviewEl.value.trim() : "";
    return customer;
  }

  function renderEvidence(items) {
    const list = document.getElementById("evidence-list");
    if (!list) return;
    list.replaceChildren();

    if (!items || !items.length) {
      const item = document.createElement("li");
      item.className = "evidence-item";
      item.textContent = "Telemetry indicators are within nominal operational thresholds.";
      list.append(item);
      return;
    }

    items.forEach(evidence => {
      const item = document.createElement("li");
      item.className = "evidence-item";

      const labelWrap = document.createElement("div");
      labelWrap.className = "evidence-label";

      const name = document.createElement("span");
      name.textContent = displayFeature(evidence.feature);
      labelWrap.append(name);

      const delta = document.createElement("div");
      const isRaises = evidence.direction === "raises";
      delta.className = `evidence-delta ${isRaises ? "raises" : "lowers"}`;
      delta.textContent = isRaises 
        ? `▲ Raises Churn (+${evidence.strength})` 
        : `▼ Lowers Churn (-${evidence.strength})`;

      item.append(labelWrap, delta);
      list.append(item);
    });
  }

  function renderCase(result) {
    const emptyResult = document.getElementById("empty-result");
    const predResult = document.getElementById("prediction-result");
    if (emptyResult) emptyResult.classList.add("hidden");
    if (predResult) predResult.classList.remove("hidden");

    const riskVal = result.risk_percent.toFixed(1);
    const riskNum = document.getElementById("risk-number");
    if (riskNum) riskNum.textContent = riskVal;

    const badge = document.getElementById("risk-badge");
    const track = document.getElementById("risk-track");
    if (track) track.style.width = `${Math.max(3, result.risk_percent)}%`;

    if (badge) {
      if (result.risk_percent >= 60) {
        badge.textContent = "CRITICAL CHURN RISK (≥60%)";
        badge.className = "risk-tag critical";
      } else if (result.risk_percent >= 40) {
        badge.textContent = "ELEVATED RISK (40-60%)";
        badge.className = "risk-tag moderate";
      } else {
        badge.textContent = "RETAINED / HEALTHY (<40%)";
        badge.className = "risk-tag healthy";
      }
    }

    const targetDef = document.getElementById("target-definition");
    if (targetDef) {
      const statusText = result.churn_status === "inactive_90_days" ? "INACTIVE INTENT (90+ days)" : "ACTIVE INTENT (<90 days)";
      targetDef.textContent = `${statusText} · Decision Threshold: 50.0% · ${result.label_name || "90-Day Repeat Order"}`;
    }

    const reviewSignal = document.getElementById("review-signal");
    if (reviewSignal) {
      if (result.review_signal && result.review_signal.sentiment !== "unavailable") {
        reviewSignal.classList.remove("hidden");
        const sentiment = result.review_signal.sentiment;
        const score = result.review_signal.positive_score_percent;
        reviewSignal.innerHTML = `<strong>Customer Review NLP:</strong> ${sentiment.toUpperCase()} (${score}% positive polarity)`;
      } else {
        reviewSignal.classList.add("hidden");
      }
    }

    const segBadge = document.getElementById("segment-badge");
    if (segBadge) {
      if (result.customer_segment) {
        segBadge.classList.remove("hidden");
        segBadge.textContent = result.customer_segment.short_name || result.customer_segment.name;
        segBadge.title = result.customer_segment.description || "Unsupervised K-Means Cluster";
        if (result.customer_segment.badge_color) {
          segBadge.style.background = `${result.customer_segment.badge_color}22`;
          segBadge.style.color = result.customer_segment.badge_color;
          segBadge.style.borderColor = `${result.customer_segment.badge_color}55`;
        }
      } else {
        segBadge.classList.add("hidden");
      }
    }

    renderEvidence(result.evidence || []);

    const modelQuality = document.getElementById("model-quality");
    if (modelQuality && result.metrics) {
      const precision = Number(result.metrics.precision).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 0 });
      const recall = Number(result.metrics.recall).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 0 });
      modelQuality.textContent =
        `Holdout Split (20k Customers): ROC-AUC ${Number(result.metrics.roc_auc).toFixed(2)} · Precision ${precision} · Recall ${recall} · F1 ${Number(result.metrics.f1).toFixed(2)} · Natural Base Rate ${result.positive_snapshot_rate_percent}%`;
    }
  }

  function clearChat() {
    history.splice(0, history.length);
    if (!chatMessages) return;
    chatMessages.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.textContent = "Ask any follow-up question regarding this customer's churn signals, delivery delays, or intervention strategy.";
    chatMessages.append(empty);
  }

  function addBubble(role, content) {
    if (!chatMessages) return;
    const empty = chatMessages.querySelector(".chat-empty");
    if (empty) empty.remove();
    const bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;
    bubble.innerHTML = renderMarkdown(content);
    chatMessages.append(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function requestRecommendation() {
    if (!latestCustomer || !latestIndustry || !recommendationCard) return;
    recommendationCard.classList.remove("hidden");
    const providerLabel = document.getElementById("recommendation-provider");
    const bodyEl = document.getElementById("recommendation-text");

    if (providerLabel) providerLabel.textContent = "Synthesizing AI Retention Playbook...";
    if (bodyEl) bodyEl.innerHTML = "<p>Analyzing customer signals against grounded service recovery protocols...</p>";

    try {
      const result = await postJson("/api/recommendations", {
        industry: latestIndustry,
        customer: latestCustomer,
        consent: consent ? consent.checked : true
      });

      if (providerLabel) providerLabel.textContent = "AI Retention Playbook & Action Plan";
      if (bodyEl) bodyEl.innerHTML = renderMarkdown(result.answer);
    } catch (error) {
      if (providerLabel) providerLabel.textContent = "AI Advisory Notice";
      if (bodyEl) bodyEl.innerHTML = `<p>${error.message}</p>`;
    }
  }

  if (form) {
    form.addEventListener("submit", async event => {
      event.preventDefault();
      clearError();
      if (analyzeButton) {
        analyzeButton.disabled = true;
        const btnText = analyzeButton.querySelector(".btn-text");
        if (btnText) btnText.textContent = "Executing Inference Pipeline...";
      }

      try {
        latestCustomer = collectCustomer();
        latestCustomer.total_historical_spend = Number(latestCustomer.average_order_value || 0) * Number(latestCustomer.orders_in_history || 1);
        latestCustomer.average_items_per_order = Number(latestCustomer.average_items_per_order || 1);
        latestCustomer.last_order_value = Number(latestCustomer.average_order_value || 0);
        latestCustomer.last_freight_value = Number(latestCustomer.last_freight_value || 0);
        latestCustomer.last_payment_installments = Number(latestCustomer.last_payment_installments || 1);
        latestCustomer.last_payment_type = latestCustomer.last_payment_type || "unknown";
        latestCustomer.last_review_score = Number(latestCustomer.average_review_score || 0);
        latestCustomer.average_delivery_duration_days = Number(latestCustomer.average_delivery_duration_days || 0);
        
        const result = await postJson("/api/predict", {
          industry: latestIndustry,
          customer: latestCustomer
        });

        renderCase(result);
        clearChat();
        if (chatInput) chatInput.disabled = false;
        if (chatSubmit) chatSubmit.disabled = false;

        if (consent && consent.checked) {
          await requestRecommendation();
        } else if (recommendationCard) {
          recommendationCard.classList.add("hidden");
        }
      } catch (error) {
        showError(error.message);
      } finally {
        if (analyzeButton) {
          analyzeButton.disabled = false;
          const btnText = analyzeButton.querySelector(".btn-text");
          if (btnText) btnText.textContent = "Execute Churn Risk Analysis";
        }
      }
    });
  }

  if (consent) {
    consent.addEventListener("change", () => {
      if (!consent.checked) {
        if (recommendationCard) recommendationCard.classList.add("hidden");
        history.splice(0, history.length);
        return;
      }
      if (latestCustomer) requestRecommendation();
    });
  }

  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      if (!latestCustomer) {
        showError("Please execute an analysis first to establish customer context.");
        return;
      }
      if (chatInput) chatInput.value = chip.dataset.prompt;
      if (chatForm) chatForm.dispatchEvent(new Event("submit"));
    });
  });

  if (chatForm) {
    chatForm.addEventListener("submit", async event => {
      event.preventDefault();
      const question = chatInput ? chatInput.value.trim() : "";
      if (!question || !latestCustomer || !latestIndustry) return;

      if (consent && !consent.checked) {
        showError("Enable AI Advisor to chat with the retention copilot.");
        return;
      }

      clearError();
      addBubble("user", question);
      if (chatInput) {
        chatInput.value = "";
        chatInput.disabled = true;
      }
      if (chatSubmit) chatSubmit.disabled = true;

      try {
        const result = await postJson("/api/chat", {
          industry: latestIndustry,
          customer: latestCustomer,
          question,
          history: history.slice(-8),
          consent: consent ? consent.checked : true
        });
        addBubble("assistant", result.answer);
        history.push({ role: "user", content: question }, { role: "assistant", content: result.answer });
      } catch (error) {
        addBubble("assistant", error.message);
      } finally {
        if (chatInput) {
          chatInput.disabled = false;
          chatInput.focus();
        }
        if (chatSubmit) chatSubmit.disabled = false;
      }
    });
  }

  // Theme Toggle (Dark & Light)
  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      const nextTheme = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      try {
        localStorage.setItem("retainx_theme", nextTheme);
      } catch (e) {}
    });
  }

  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebarPreference = (() => { try { return localStorage.getItem("retainx_sidebar_collapsed") === "true"; } catch (e) { return false; } })();
  if (sidebarPreference && window.matchMedia("(min-width: 641px)").matches) document.body.classList.add("sidebar-collapsed");
  if (sidebarToggle) {
    const updateSidebarButton = () => {
      const collapsed = document.body.classList.contains("sidebar-collapsed");
      sidebarToggle.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
      sidebarToggle.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
    };
    updateSidebarButton();
    sidebarToggle.addEventListener("click", () => {
      document.body.classList.toggle("sidebar-collapsed");
      updateSidebarButton();
      try { localStorage.setItem("retainx_sidebar_collapsed", document.body.classList.contains("sidebar-collapsed") ? "true" : "false"); } catch (e) {}
    });
  }

  // --- Multi-View Navigation: Simulator, Batch Cohort & Analytics ---
  const viewSimulator = document.getElementById("view-simulator");
  const viewBatch = document.getElementById("view-batch");
  const viewAnalytics = document.getElementById("view-analytics");
  const navBtnSimulator = document.getElementById("nav-btn-simulator");
  const navBtnBatch = document.getElementById("nav-btn-batch");
  const navBtnAnalytics = document.getElementById("nav-btn-analytics");
  const analyticsBackBtn = document.getElementById("analytics-back-btn");

  function switchView(target) {
    if (viewSimulator) viewSimulator.classList.add("hidden");
    if (viewBatch) viewBatch.classList.add("hidden");
    if (viewAnalytics) viewAnalytics.classList.add("hidden");

    if (navBtnSimulator) navBtnSimulator.classList.remove("active");
    if (navBtnBatch) navBtnBatch.classList.remove("active");
    if (navBtnAnalytics) navBtnAnalytics.classList.remove("active");

    if (target === "analytics") {
      if (viewAnalytics) viewAnalytics.classList.remove("hidden");
      if (navBtnAnalytics) navBtnAnalytics.classList.add("active");
      viewAnalytics?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (target === "batch") {
      if (viewBatch) viewBatch.classList.remove("hidden");
      if (navBtnBatch) navBtnBatch.classList.add("active");
      viewBatch?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      if (viewSimulator) viewSimulator.classList.remove("hidden");
      if (navBtnSimulator) navBtnSimulator.classList.add("active");
      viewSimulator?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  if (navBtnSimulator) navBtnSimulator.addEventListener("click", () => switchView("simulator"));
  if (navBtnBatch) navBtnBatch.addEventListener("click", () => switchView("batch"));
  if (navBtnAnalytics) navBtnAnalytics.addEventListener("click", () => switchView("analytics"));
  if (analyticsBackBtn) analyticsBackBtn.addEventListener("click", () => switchView("simulator"));

  // --- Enterprise Batch Prediction Logic ---
  let selectedBatchFile = null;
  const batchDropzone = document.getElementById("batch-dropzone");
  const batchFileInput = document.getElementById("batch-file-input");
  const batchBrowseBtn = document.getElementById("batch-browse-btn");
  const selectedFileBar = document.getElementById("selected-file-bar");
  const selectedFileName = document.getElementById("selected-file-name");
  const selectedFileSize = document.getElementById("selected-file-size");
  const batchRemoveFileBtn = document.getElementById("batch-remove-file-btn");
  const batchExecuteBtn = document.getElementById("batch-execute-btn");
  const batchLoading = document.getElementById("batch-loading");
  const batchResults = document.getElementById("batch-results");
  const batchExportBtn = document.getElementById("batch-export-btn");

  function handleBatchFileSelect(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      alert("Please upload a CSV file (.csv format).");
      return;
    }
    selectedBatchFile = file;
    if (selectedFileName) selectedFileName.textContent = file.name;
    const kb = (file.size / 1024).toFixed(1);
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    if (selectedFileSize) selectedFileSize.textContent = file.size > 1024 * 1024 ? `${mb} MB` : `${kb} KB`;
    if (selectedFileBar) selectedFileBar.classList.remove("hidden");
  }

  if (batchBrowseBtn && batchFileInput) {
    batchBrowseBtn.addEventListener("click", () => batchFileInput.click());
  }

  if (batchDropzone) {
    batchDropzone.addEventListener("click", (e) => {
      if (e.target !== batchBrowseBtn && batchFileInput) {
        batchFileInput.click();
      }
    });

    ["dragenter", "dragover"].forEach(eventName => {
      batchDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        batchDropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      batchDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        batchDropzone.classList.remove("dragover");
      });
    });

    batchDropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length) {
        handleBatchFileSelect(dt.files[0]);
      }
    });
  }

  if (batchFileInput) {
    batchFileInput.addEventListener("change", () => {
      if (batchFileInput.files && batchFileInput.files.length) {
        handleBatchFileSelect(batchFileInput.files[0]);
      }
    });
  }

  if (batchRemoveFileBtn) {
    batchRemoveFileBtn.addEventListener("click", () => {
      selectedBatchFile = null;
      if (batchFileInput) batchFileInput.value = "";
      if (selectedFileBar) selectedFileBar.classList.add("hidden");
      if (batchResults) batchResults.classList.add("hidden");
    });
  }

  if (batchExecuteBtn) {
    batchExecuteBtn.addEventListener("click", async () => {
      if (!selectedBatchFile) {
        alert("Please select or drop a CSV file first.");
        return;
      }

      batchExecuteBtn.disabled = true;
      if (batchLoading) batchLoading.classList.remove("hidden");
      if (batchResults) batchResults.classList.add("hidden");

      const formData = new FormData();
      formData.append("file", selectedBatchFile);

      try {
        const response = await fetch("/api/batch-predict", {
          method: "POST",
          body: formData,
        });

        const contentType = response.headers.get("content-type") || "";
        let data;
        if (contentType.includes("application/json")) {
          data = await response.json();
        } else {
          const text = await response.text();
          throw new Error(`Server returned HTTP ${response.status} (${response.statusText || "non-JSON"}). Check dataset structure.`);
        }

        if (!response.ok) {
          throw new Error(data.error || "Batch prediction failed.");
        }

        renderBatchResults(data);
      } catch (err) {
        alert("Batch Execution Notice: " + err.message);
      } finally {
        batchExecuteBtn.disabled = false;
        if (batchLoading) batchLoading.classList.add("hidden");
      }
    });
  }

  function renderBatchResults(data) {
    if (!batchResults) return;
    batchResults.classList.remove("hidden");

    // KPI Metrics
    const kpiTot = document.getElementById("batch-kpi-total");
    const kpiRet = document.getElementById("batch-kpi-retain");
    const kpiRetPct = document.getElementById("batch-kpi-retain-pct");
    const kpiChu = document.getElementById("batch-kpi-churn");
    const kpiChuPct = document.getElementById("batch-kpi-churn-pct");
    const kpiAvgR = document.getElementById("batch-kpi-avg-risk");

    if (kpiTot) kpiTot.textContent = Number(data.total_customers).toLocaleString();
    if (kpiRet) kpiRet.textContent = Number(data.summary.will_retain).toLocaleString();
    if (kpiRetPct) kpiRetPct.textContent = `(${data.summary.retain_percent}%)`;
    if (kpiChu) kpiChu.textContent = Number(data.summary.will_churn).toLocaleString();
    if (kpiChuPct) kpiChuPct.textContent = `(${data.summary.churn_percent}%)`;
    if (kpiAvgR) kpiAvgR.textContent = data.summary.avg_risk_percent.toFixed(1);

    // Risk Bands
    const rb = data.risk_distribution;
    if (rb) {
      const bLowC = document.getElementById("band-low-count");
      const bLowF = document.getElementById("band-low-fill");
      const bLowP = document.getElementById("band-low-pct");
      if (bLowC) bLowC.textContent = Number(rb.low_risk.count).toLocaleString();
      if (bLowF) bLowF.style.width = `${rb.low_risk.percent}%`;
      if (bLowP) bLowP.textContent = `${rb.low_risk.percent}%`;

      const bModC = document.getElementById("band-mod-count");
      const bModF = document.getElementById("band-mod-fill");
      const bModP = document.getElementById("band-mod-pct");
      if (bModC) bModC.textContent = Number(rb.moderate_risk.count).toLocaleString();
      if (bModF) bModF.style.width = `${rb.moderate_risk.percent}%`;
      if (bModP) bModP.textContent = `${rb.moderate_risk.percent}%`;

      const bHigC = document.getElementById("band-high-count");
      const bHigF = document.getElementById("band-high-fill");
      const bHigP = document.getElementById("band-high-pct");
      if (bHigC) bHigC.textContent = Number(rb.high_risk.count).toLocaleString();
      if (bHigF) bHigF.style.width = `${rb.high_risk.percent}%`;
      if (bHigP) bHigP.textContent = `${rb.high_risk.percent}%`;

      const bCriC = document.getElementById("band-crit-count");
      const bCriF = document.getElementById("band-crit-fill");
      const bCriP = document.getElementById("band-crit-pct");
      if (bCriC) bCriC.textContent = Number(rb.critical_risk.count).toLocaleString();
      if (bCriF) bCriF.style.width = `${rb.critical_risk.percent}%`;
      if (bCriP) bCriP.textContent = `${rb.critical_risk.percent}%`;
    }

    // K-Means Segments
    const segGrid = document.getElementById("batch-segments-grid");
    if (segGrid && data.segment_distribution) {
      segGrid.replaceChildren();

      Object.values(data.segment_distribution).forEach(seg => {
        const card = document.createElement("div");
        card.className = "segment-card";

        card.innerHTML = `
          <div>
            <div class="segment-card-header">
              <span class="segment-pill" style="background:${seg.badge_color}">${seg.short_name}</span>
              <span class="segment-count-badge">${Number(seg.count).toLocaleString()} (${seg.percent}%)</span>
            </div>
            <h4>${seg.name}</h4>
            <p class="segment-desc">${getSegmentDescription(seg.cluster_id)}</p>
          </div>
          <div class="segment-action-box">
            <strong>Strategic Playbook:</strong>
            ${seg.action}
          </div>
        `;
        segGrid.appendChild(card);
      });
    }

    // Top At-Risk Customers Table
    const tbody = document.getElementById("batch-table-body");
    if (tbody && data.top_at_risk_customers) {
      tbody.replaceChildren();

      data.top_at_risk_customers.forEach(cust => {
        const tr = document.createElement("tr");
        const riskClass = cust.risk_percent >= 80 ? "critical" : cust.risk_percent >= 60 ? "high" : "low";

        tr.innerHTML = `
          <td><strong>${cust.customer_id}</strong></td>
          <td>${cust.state}</td>
          <td><span class="risk-cell-badge ${riskClass}">${cust.risk_percent}%</span></td>
          <td><strong>${cust.predicted_status}</strong></td>
          <td><span class="segment-pill" style="background:${cust.badge_color}; color:#fff; font-size:10.5px; padding:2px 7px;">${cust.segment_short}</span></td>
          <td>${cust.days_since_last_order} days</td>
          <td>₹${Number(cust.total_spend).toLocaleString()}</td>
          <td>${Number(cust.rating).toFixed(1)}★</td>
          <td>${cust.delivery_delay > 0 ? '+' + Number(cust.delivery_delay).toFixed(1) + 'd' : 'On-Time'}</td>
          <td style="max-width:260px; white-space:normal; font-size:11.5px; color:#cbd5e1;">${cust.action}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    batchResults.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function getSegmentDescription(cid) {
    const descs = {
      0: "Healthy recurring purchasers with low dormancy (~17 days) and steady basket volumes.",
      1: "Customers experiencing fulfillment friction (+6.4 days SLA breach) and long transit delays.",
      2: "Lapsed shoppers crossing 60-90+ days inactivity requiring immediate win-back coupons.",
      3: "Top-tier enterprise VIPs with high lifetime spend (₹8.5L+) and premium basket values."
    };
    return descs[cid] || "Discovered behavioral customer cluster.";
  }

  // Export Scored Batch CSV
  if (batchExportBtn) {
    batchExportBtn.addEventListener("click", async () => {
      if (!selectedBatchFile) {
        alert("Please select a CSV file first.");
        return;
      }

      batchExportBtn.disabled = true;
      const originalText = batchExportBtn.innerHTML;
      batchExportBtn.innerHTML = "<span>Generating Scored CSV...</span>";

      const formData = new FormData();
      formData.append("file", selectedBatchFile);

      try {
        const response = await fetch("/api/batch-predict/export", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const contentType = response.headers.get("content-type") || "";
          let errDetail = `CSV Export Failed (${response.status})`;
          if (contentType.includes("application/json")) {
            const errJson = await response.json();
            errDetail = errJson.error || errDetail;
          }
          throw new Error(errDetail);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = "retainx_batch_predictions_scored.csv";
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        alert("Export Error: " + err.message);
      } finally {
        batchExportBtn.disabled = false;
        batchExportBtn.innerHTML = originalText;
      }
    });
  }

  // Studio Sub-Tabs (ROC, CM, Features, Segmentation, NLP, Benchmark)
  const studioTabs = document.querySelectorAll(".studio-tab-btn");
  const studioPanels = document.querySelectorAll(".studio-panel");

  studioTabs.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.dataset.tab;
      studioTabs.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      studioPanels.forEach(panel => {
        if (panel.id === `tab-panel-${tabId}`) {
          panel.classList.remove("hidden");
        } else {
          panel.classList.add("hidden");
        }
      });
    });
  });

  // Graph Image Lightbox Modal
  const lightboxModal = document.getElementById("graph-lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxClose = document.getElementById("lightbox-close");
  const zoomableWrappers = document.querySelectorAll(".graph-image-wrapper");

  function openLightbox(src, alt) {
    if (!lightboxModal || !lightboxImg) return;
    lightboxImg.src = src;
    lightboxImg.alt = alt || "Graph preview";
    lightboxModal.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    if (!lightboxModal) return;
    lightboxModal.classList.remove("open");
    document.body.style.overflow = "";
  }

  zoomableWrappers.forEach(wrapper => {
    wrapper.addEventListener("click", () => {
      const img = wrapper.querySelector("img");
      if (img) openLightbox(img.src, img.alt);
    });
  });

  if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
  if (lightboxModal) {
    lightboxModal.addEventListener("click", (e) => {
      if (e.target === lightboxModal) closeLightbox();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightboxModal && lightboxModal.classList.contains("open")) {
      closeLightbox();
    }
  });

  // Safe initialize
  try {
    setIndustry("ecommerce");
  } catch (err) {
    console.warn("Init notice:", err);
  }
})();
