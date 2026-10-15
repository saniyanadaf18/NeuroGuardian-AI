/* =====================================================
   NeuroGuardian – shared core (ng-core.js)
   Put this file in the SAME folder as all the .html files.
   Add this line in <head> of every module (except nothing else needed):
       <script src="ng-core.js"></script>
   What it does:
     1. Login guard (redirects to Login.html if no session)
     2. Fixes wrong/lowercase/-1/-2 links to the real page names
     3. NG.saveReport() -> feeds report_history.html
     4. Floating "Dashboard" button on mobile if page has none
   ===================================================== */
(function () {
  "use strict";

  var PAGES = [
    "Dashboard.html", "Login.html", "Register.html", "ai_chat.html",
    "blood_report.html", "brain_disease.html", "ct_mri.html",
    "eye_neuro_screening.html", "migraine_analysis.html",
    "report_history.html", "vision.html"
  ];
  var PUBLIC = ["login.html", "register.html"];

  /* ---------- safe storage ---------- */
  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); return true; } catch (e) { return false; } }
  function ssGet(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
  function parse(s, fallback) { try { var v = JSON.parse(s); return v == null ? fallback : v; } catch (e) { return fallback; } }

  /* ---------- page name helpers ---------- */
  function canonical(href) {
    if (!href) return null;
    var clean = href.split("#")[0].split("?")[0];
    if (/^(https?:|mailto:|tel:|javascript:)/i.test(clean)) return null;
    var file = clean.split("/").pop();
    if (!/\.html$/i.test(file)) return null;
    file = file.replace(/-\d+(?=\.html$)/i, "").toLowerCase();
    if (file === "migraine_analy.html") file = "migraine_analysis.html";
    for (var i = 0; i < PAGES.length; i++) {
      if (PAGES[i].toLowerCase() === file) return PAGES[i];
    }
    return null;
  }
  function currentPage() {
    return (canonical(location.pathname) || "").toLowerCase();
  }

  /* ---------- session ---------- */
  function getSession() {
    return parse(lsGet("neuroguardianSession"), null) ||
           parse(ssGet("neuroguardianSession"), null);
  }
  function logout() {
    try { localStorage.removeItem("neuroguardianSession"); } catch (e) {}
    try { sessionStorage.removeItem("neuroguardianSession"); } catch (e) {}
    location.href = "Login.html";
  }
  function storageWorks() {
    try {
      localStorage.setItem("__ng_test", "1");
      localStorage.removeItem("__ng_test");
      return true;
    } catch (e) { return false; }
  }
  function requireLogin() {
    if (PUBLIC.indexOf(currentPage()) !== -1) return;
    /* Pages opened straight from phone/computer storage (file://) cannot
       keep a login between pages, so the check is skipped there.
       It works normally once the project is hosted (https://). */
    if (location.protocol === "file:") return;
    /* If the browser blocks storage (private mode), do not lock the user out */
    if (!storageWorks()) return;
    if (!getSession()) location.replace("Login.html");
  }

  /* ---------- reports (read by report_history.html) ---------- */
  function saveReport(r) {
    var list = parse(lsGet("ng_reports"), []);
    if (!Array.isArray(list)) list = [];
    var s = getSession();
    var entry = {
      id: r.id || (Date.now().toString() + Math.floor(Math.random() * 1000)),
      module: r.module,                       // blood|brain|ctmri|migraine|vision|eyescan|symptom
      date: r.date || new Date().toISOString(),
      tier: r.tier || "neutral",              // green|yellow|red|neutral
      title: r.title || "Report",
      summary: r.summary || "",
      details: r.details || {},
      user: s && s.email ? s.email : "unknown"
    };
    list = list.filter(function (x) { return x.id !== entry.id; });
    list.push(entry);
    lsSet("ng_reports", JSON.stringify(list));
    return entry;
  }

  /* ---------- fix links ---------- */
  function fixLinks() {
    var links = document.querySelectorAll("a[href]");
    for (var i = 0; i < links.length; i++) {
      var raw = links[i].getAttribute("href");
      var fixed = canonical(raw);
      if (fixed) {
        var tail = raw.indexOf("#") > -1 ? raw.slice(raw.indexOf("#")) : "";
        links[i].setAttribute("href", fixed + tail);
      }
    }
  }

  /* ---------- floating home button (mobile friendly) ---------- */
  function addHomeButton() {
    var p = currentPage();
    if (p === "dashboard.html" || PUBLIC.indexOf(p) !== -1) return;
    if (document.querySelector('a[href^="Dashboard.html"]')) return;
    if (document.body && document.body.getAttribute("data-ng-nav") === "off") return;
    var a = document.createElement("a");
    a.href = "Dashboard.html";
    a.setAttribute("aria-label", "Back to Dashboard");
    a.textContent = "\uD83C\uDFE0";
    a.style.cssText =
      "position:fixed;left:14px;bottom:calc(14px + env(safe-area-inset-bottom,0px));" +
      "z-index:99999;width:48px;height:48px;border-radius:50%;display:flex;" +
      "align-items:center;justify-content:center;font-size:22px;text-decoration:none;" +
      "background:#2563eb;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.3);";
    document.body.appendChild(a);
  }

  /* ---------- run ---------- */
  requireLogin();
  document.addEventListener("DOMContentLoaded", function () {
    fixLinks();
    addHomeButton();
  });

  window.NG = {
    getSession: getSession,
    logout: logout,
    saveReport: saveReport,
    go: function (page) { location.href = canonical(page) || page; }
  };
})();
