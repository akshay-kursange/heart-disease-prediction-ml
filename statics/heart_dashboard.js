const dashboardFields = [
  "age",
  "sex",
  "cp",
  "trestbps",
  "chol",
  "fbs",
  "restecg",
  "thalach",
  "exang",
  "oldpeak",
  "slope",
  "ca",
  "thal",
];

function getDashboardInput(name) {
  return document.getElementById(`hd-${name}`);
}

function collectDashboardPayload() {
  return dashboardFields.reduce((payload, field) => {
    const input = getDashboardInput(field);
    payload[field] = input ? Number(input.value) : 0;
    return payload;
  }, {});
}

function fillDashboardSample() {
  const rows = window.heartDashboardRows || [];
  if (!rows.length) return;

  const row = rows[Math.floor(Math.random() * rows.length)];
  dashboardFields.forEach((field) => {
    const input = getDashboardInput(field);
    if (input && row.values[field] !== undefined) {
      input.value = row.values[field];
    }
  });
}

function updateDashboardChart(featureImportance) {
  const chart = document.getElementById("hd-chart");
  const entries = Object.entries(featureImportance || {}).sort((a, b) => a[1] - b[1]);
  const max = Math.max(...entries.map((entry) => entry[1]), 1);

  chart.innerHTML = entries
    .map(([label, value]) => {
      const percent = (value / max) * 100;
      return `
        <div class="hd-bar-row">
          <span>${label}</span>
          <div class="hd-bar-track">
            <div class="hd-bar-fill" style="width: ${percent}%"></div>
          </div>
          <em>${value.toFixed(3)}</em>
        </div>
      `;
    })
    .join("");
}

function setDashboardResult(data) {
  const result = document.getElementById("hd-result");
  const resultText = document.getElementById("hd-result-text");
  const probability = document.getElementById("hd-probability");

  result.classList.toggle("is-risk", data.prediction === "Positive");
  resultText.textContent = data.prediction;
  probability.textContent =
    typeof data.probability === "number" ? `${data.probability.toFixed(1)}%` : "--";
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("heart-dashboard-form");
  const sampleButton = document.getElementById("hd-sample-btn");
  const submitButton = form.querySelector(".hd-predict");
  const buttonText = submitButton.querySelector(".hd-btn-text");

  sampleButton.addEventListener("click", fillDashboardSample);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submitButton.classList.add("is-loading");
    buttonText.textContent = "PREDICTING";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectDashboardPayload()),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setDashboardResult(data);
      updateDashboardChart(data.feature_importance);
    } catch (error) {
      setDashboardResult({ prediction: "Error", probability: null });
      document.getElementById("hd-probability").textContent = error.message;
    } finally {
      submitButton.classList.remove("is-loading");
      buttonText.textContent = "PREDICT";
    }
  });
});
