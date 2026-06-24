const VIEWS = {
  expected: {
    endpoint: "/api/expected-stats",
    title: "Expected Stats Leaderboard",
    headers: ["Player","PA","xBA","xSLG","xwOBA","xOBP","wOBA","BA","Edge"],
    keys: ["player","pa","xba","xslg","xwoba","xobp","woba","ba","edge"],
  },
  exitvelo: {
    endpoint: "/api/exit-velo",
    title: "Exit Velo & Barrel Leaderboard",
    headers: ["Player","Avg EV","Barrel%","Hard Hit%","Avg Dist","Avg HR Dist"],
    keys: ["player","avg_exit_velo","barrel_pct","hard_hit_pct","avg_distance","avg_hr_distance"],
  },
  percentile: {
    endpoint: "/api/percentile-ranks",
    title: "Percentile Ranks",
    headers: ["Player","Exit Velo","Hard Hit","Barrel","Whiff%","Sprint Spd"],
    keys: ["player","exit_velocity","hard_hit_rate","barrel_batted_rate","whiff_percent","sprint_speed"],
  }
};

let currentView = "expected";

async function loadKPIs() {
  try {
    const res = await fetch("/api/kpis");
    const json = await res.json();
    if (json.status !== "ok") return;
    document.getElementById("kpis").innerHTML = `
      <div class="kpi"><span>Avg Exit Velo</span><strong>${json.avg_exit_velo} mph</strong></div>
      <div class="kpi"><span>Avg Barrel%</span><strong>${json.avg_barrel_rate}%</strong></div>
      <div class="kpi"><span>Avg Hard Hit%</span><strong>${json.avg_hard_hit}%</strong></div>
      <div class="kpi"><span>Avg xwOBA</span><strong>${json.avg_xwoba}</strong></div>
      <div class="kpi"><span>Top Edge Player</span><strong>${json.top_positive_edge}</strong></div>
    `;
    document.getElementById("subtitle").textContent = `Live ${json.year} Statcast · Updates hourly`;
  } catch(e) {
    document.getElementById("subtitle").textContent = "Error loading KPIs";
  }
}

async function loadTable(view) {
  const config = VIEWS[view];
  document.getElementById("tableTitle").textContent = config.title;
  document.getElementById("tableHead").innerHTML =
    "<tr>" + config.headers.map(h => `<th>${h}</th>`).join("") + "</tr>";
  document.getElementById("tableBody").innerHTML =
    `<tr><td colspan="${config.headers.length}" class="loading">Fetching from Baseball Savant...</td></tr>`;
  try {
    const res = await fetch(config.endpoint);
    const json = await res.json();
    if (json.status !== "ok") {
      document.getElementById("tableBody").innerHTML =
        `<tr><td colspan="${config.headers.length}" class="loading error">Error: ${json.message}</td></tr>`;
      return;
    }
    document.getElementById("tableBody").innerHTML = json.data.map(row => {
      const cells = config.keys.map(k => {
        let val = row[k] ?? "—";
        if (k === "edge" && typeof val === "number") {
          const cls = val > 0 ? "up" : val < -0.01 ? "down" : "";
          return `<td class="${cls}">${val > 0 ? "+" : ""}${val}</td>`;
        }
        return `<td>${val}</td>`;
      }).join("");
      return `<tr>${cells}</tr>`;
    }).join("");
    document.getElementById("dataSource").textContent = `Live · ${json.source}`;
  } catch(e) {
    document.getElementById("tableBody").innerHTML =
      `<tr><td colspan="${config.headers.length}" class="loading error">Error: ${e.message}</td></tr>`;
  }
}

function switchView() {
  currentView = document.getElementById("viewSelect").value;
  loadTable(currentView);
}

loadKPIs();
loadTable(currentView);
