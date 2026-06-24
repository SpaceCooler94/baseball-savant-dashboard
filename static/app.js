async function loadKPIs() {
  const res = await fetch("/api/kpis");
  const data = await res.json();
  const el = document.getElementById("kpis");
  el.innerHTML = `
    <div class="kpi"><span>Avg EV</span><strong>${data.avg_exit_velo} mph</strong></div>
    <div class="kpi"><span>Hard Hit</span><strong>${data.hard_hit_rate}%</strong></div>
    <div class="kpi"><span>Barrel</span><strong>${data.barrel_rate}%</strong></div>
    <div class="kpi"><span>Whiff</span><strong>${data.whiff_rate}%</strong></div>
    <div class="kpi"><span>Run Index</span><strong>${data.expected_run_index}</strong></div>
  `;
}

async function loadPlayers() {
  const res = await fetch("/api/players");
  const rows = await res.json();
  const tbody = document.getElementById("playerTable");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.player}</td>
      <td>${Number(r.xba).toFixed(3)}</td>
      <td>${Number(r.xslg).toFixed(3)}</td>
      <td>${r.barrel_pct}%</td>
      <td>${r.sweet_spot_pct}%</td>
      <td>${r.edge > 0 ? "+" : ""}${r.edge}%</td>
    </tr>
  `).join("");
}

loadKPIs();
loadPlayers();
