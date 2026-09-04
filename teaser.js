/* v1 project page — interactive overview figure.
   Replays one reasoning trace under two decoders, one generating step at a time:
   a standard text-only head, and v1's point-and-copy head. */
(() => {
  "use strict";

  const COLS = 7, ROWS = 8;               // patch grid of the input diagram
  const cellName = (c, r) => `patch (${c},${r})`;

  // Candidate helper: [label, prob] for text tokens, {cell:[c,r], p} for visual tokens.
  const T = (label, p) => ({ kind: "txt", label, p });
  const P = (c, r, p) => ({ kind: "img", cell: [c, r], label: cellName(c, r), p });

  // Both panels use six steps so they advance in lockstep.
  const STD = [
    { pre: "… we can use the Pythagorean", key: "Theorem", tail: "…",
      cands: [T("Theorem", .61), T("theorem", .12), T("formula", .08), T("identity", .05), T("relation", .04), T("triple", .03), T("equation", .02), T("rule", .02)] },
    { pre: "In this case, it appears that ∠RTS is a", key: "right", tail: "angle.", cls: "bad",
      cands: [T("right", .47), T("straight", .11), T("acute", .10), T("obtuse", .08), T("90°", .07), T("central", .06), T("base", .04), T("small", .03)] },
    { pre: "Applying the Pythagorean Theorem, we get: (2z − 15)² = 7² + 9² = 49 + 81 =", key: "130", tail: "",
      cands: [T("130", .83), T("120", .05), T("131", .03), T("129", .02), T("13", .02), T("140", .02), T("128", .01), T("100", .01)] },
    { pre: "→ 2z − 15 ≈", key: "11.4", tail: "",
      cands: [T("11.4", .52), T("11.40", .14), T("11.5", .11), T("11", .09), T("11.3", .06), T("12", .04), T("10", .02), T("11.45", .02)] },
    { pre: "Rounding to the nearest integer, we get", key: "z = 13", tail: ".", cls: "bad",
      cands: [T("z=13", .58), T("z", .12), T("13", .10), T("z≈13", .08), T("z=12", .05), T("2z", .03), T("z=14", .02), T("26", .02)] },
    { pre: "The closest answer choice is", key: "(C) 12", tail: ", so that is…", cls: "ans",
      cands: [T("(C)", .70), T("C", .12), T("(B)", .07), T("(D)", .05), T("12", .03), T("(A)", .02), T("option", .01), T("choice", .00)] },
  ];
  STD.verdict = { text: "answer (C) 12 — but ∠RTS is not a right angle: the model never re-checked the diagram.", cls: "bad" };

  // Visual-token slots shown in the v1 chart (fixed order so bars animate in place).
  const S_ANG = [1, 4], T_ANG = [5, 6], NINE = [6, 3], LBL = [1, 2];
  const V1 = [
    { pre: "I’ve got a triangle diagram here. … according to", key: P(...S_ANG), tail: ",",
      cands: [T("the", .08), T("angle", .06), T("right", .05), T("side", .03), T("triangle", .02), P(...S_ANG, .48), P(...T_ANG, .14), P(...NINE, .09), P(...LBL, .05)] },
    { pre: "", key: P(...T_ANG), tail: ", and",
      cands: [T("angle", .07), T("and", .06), T("the", .04), T("side", .03), T("vertex", .02), P(...S_ANG, .10), P(...T_ANG, .55), P(...NINE, .09), P(...LBL, .04)] },
    { pre: "", key: P(...NINE), tail: "…",
      cands: [T("side", .06), T("length", .05), T("the", .04), T("angle", .03), T("9", .03), P(...S_ANG, .10), P(...T_ANG, .12), P(...NINE, .51), P(...LBL, .06)] },
    { pre: "we can notice that triangle RST is", key: "isosceles", tail: ", with RS = RT.", cls: "ok",
      cands: [T("isosceles", .66), T("a", .09), T("right", .06), T("equilateral", .04), T("scalene", .02), P(...S_ANG, .05), P(...T_ANG, .04), P(...NINE, .03), P(...LBL, .01)] },
    { pre: "Thus, 2z − 15 = 9 → 2z = 24 →", key: "z = 12", tail: ".", cls: "ok",
      cands: [T("z=12", .81), T("z", .07), T("12", .05), T("z=24", .02), T("2z", .01), P(...S_ANG, .02), P(...T_ANG, .01), P(...NINE, .01), P(...LBL, .00)] },
    { pre: "All in all, the answer is", key: "(C) z = 12", tail: ".", cls: "ans ok",
      cands: [T("(C)", .78), T("C", .09), T("(B)", .04), T("12", .03), T("option", .02), P(...S_ANG, .02), P(...T_ANG, .01), P(...NINE, .01), P(...LBL, .00)] },
  ];
  V1.verdict = { text: "answer (C) z = 12 — the matching angle marks at S and T were copied into the trace before the claim RS = RT.", cls: "ok" };

  const N = STD.length;
  const STEP_MS = 1900;
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const fig = document.getElementById("fig");
  if (!fig) return;
  const $ = (sel, root = fig) => root.querySelector(sel);

  /* ---------- input diagram cells ---------- */
  const cellsEl = document.getElementById("fig-cells");
  const cellEl = {};
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    const d = document.createElement("div");
    d.className = "cell"; d.dataset.cell = `${c},${r}`;
    cellsEl.appendChild(d); cellEl[`${c},${r}`] = d;
  }
  const setCell = (cell, cls, on) => { const el = cellEl[cell.join(",")]; if (el) el.classList.toggle(cls, on); };
  const clearCells = (cls) => Object.values(cellEl).forEach(el => el.classList.remove(cls));
  let hotCount = 0;
  const hot = (cell, on) => {
    setCell(cell, "hot", on);
    hotCount = Math.max(0, hotCount + (on ? 1 : -1));
    cellsEl.classList.toggle("focused", hotCount > 0);
  };

  /* ---------- panels ---------- */
  const panels = {
    std: { steps: STD, root: $(".panel-std"), slots: 8, imgSlots: 0 },
    v1:  { steps: V1,  root: $(".panel-v1"),  slots: 9, imgSlots: 4 },
  };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const CW = 360, CH = 150, BASE = 108, TOP = 10, LABEL_Y = 122;

  function buildChart(p) {
    const svg = $(".bars", p.root);
    const gap = 6, w = (CW - 12) / p.slots;
    p.bars = [];
    const axis = document.createElementNS(SVG_NS, "line");
    axis.setAttribute("x1", 6); axis.setAttribute("x2", CW - 6); axis.setAttribute("y1", BASE); axis.setAttribute("y2", BASE);
    axis.setAttribute("class", "axis"); svg.appendChild(axis);
    if (p.imgSlots) {
      const x = 6 + (p.slots - p.imgSlots) * w;
      const sep = document.createElementNS(SVG_NS, "line");
      sep.setAttribute("x1", x); sep.setAttribute("x2", x); sep.setAttribute("y1", TOP); sep.setAttribute("y2", CH - 4);
      sep.setAttribute("class", "sep"); svg.appendChild(sep);
    }
    for (let i = 0; i < p.slots; i++) {
      const isImg = i >= p.slots - p.imgSlots;
      const x = 6 + i * w + gap / 2, bw = w - gap;
      const g = document.createElementNS(SVG_NS, "g");
      g.setAttribute("class", "bar " + (isImg ? "bar-img" : "bar-txt"));
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", x); rect.setAttribute("width", bw); rect.setAttribute("y", BASE); rect.setAttribute("height", 0); rect.setAttribute("rx", 2);
      g.appendChild(rect);
      let label;
      if (isImg) {
        // nested svg with a viewBox crops the diagram to one patch
        label = document.createElementNS(SVG_NS, "svg");
        const s = Math.min(bw, 27);
        label.setAttribute("x", x + (bw - s) / 2); label.setAttribute("y", BASE + 8); label.setAttribute("width", s); label.setAttribute("height", s);
        label.setAttribute("preserveAspectRatio", "none");
        const img = document.createElementNS(SVG_NS, "image");
        img.setAttribute("href", "assets/diagram_rst.png"); img.setAttribute("width", 584); img.setAttribute("height", 672);
        label.appendChild(img);
        const frame = document.createElementNS(SVG_NS, "rect");
        frame.setAttribute("x", x + (bw - s) / 2); frame.setAttribute("y", BASE + 8); frame.setAttribute("width", s); frame.setAttribute("height", s);
        frame.setAttribute("class", "thumb-frame"); g.appendChild(frame);
        g.addEventListener("mouseenter", () => { if (g._cell) hot(g._cell, true); });
        g.addEventListener("mouseleave", () => { if (g._cell) hot(g._cell, false); });
      } else {
        label = document.createElementNS(SVG_NS, "text");
        label.setAttribute("x", x + bw / 2); label.setAttribute("y", LABEL_Y);
        label.setAttribute("transform", `rotate(-32 ${x + bw / 2} ${LABEL_Y})`);
        label.setAttribute("class", "bar-label");
      }
      g.appendChild(label);
      svg.appendChild(g);
      p.bars.push({ g, rect, label, isImg });
    }
  }

  function renderChart(p, step) {
    const out = $(".chart-out", p.root);
    if (!step) {
      p.bars.forEach(b => { b.rect.setAttribute("y", BASE); b.rect.setAttribute("height", 0); b.g.classList.remove("argmax"); });
      out.textContent = "";
      return;
    }
    const max = Math.max(...step.cands.map(c => c.p));
    step.cands.forEach((c, i) => {
      const b = p.bars[i]; if (!b) return;
      const h = Math.max(2, (c.p / max) * (BASE - TOP));
      b.rect.setAttribute("y", BASE - h); b.rect.setAttribute("height", h);
      const isMax = c.p === max;
      b.g.classList.toggle("argmax", isMax);
      if (b.isImg) {
        const [cc, rr] = c.cell;
        b.label.setAttribute("viewBox", `${cc * 584 / COLS} ${rr * 672 / ROWS} ${584 / COLS} ${672 / ROWS}`);
        b.g._cell = c.cell;
      } else {
        b.label.textContent = c.label;
      }
    });
    const key = step.key;
    out.innerHTML = typeof key === "string"
      ? `&rarr; output the text token <b>${key}</b>`
      : `&rarr; copy the input visual token <b>${key.label}</b>`;
  }

  function chip(cell) {
    const [c, r] = cell;
    const s = document.createElement("span");
    s.className = "pchip"; s.tabIndex = 0;
    s.style.setProperty("--c", c); s.style.setProperty("--r", r);
    s.setAttribute("role", "img"); s.setAttribute("aria-label", "copied " + cellName(c, r));
    const on = () => hot(cell, true), off = () => hot(cell, false);
    s.addEventListener("mouseenter", on); s.addEventListener("mouseleave", off);
    s.addEventListener("focus", on); s.addEventListener("blur", off);
    return s;
  }

  function renderTrace(p, t) {
    const el = $(".trace", p.root);
    el.innerHTML = "";
    if (t === 0) { el.innerHTML = '<span class="trace-empty">&hellip;</span>'; return; }
    for (let i = 0; i < t; i++) {
      const s = p.steps[i];
      const span = document.createElement("span");
      span.className = "trace-step" + (i === t - 1 ? " latest" : "");
      if (s.pre) span.append(s.pre + " ");
      if (typeof s.key === "string") {
        const k = document.createElement("mark");
        k.className = "key " + (s.cls || "");
        k.textContent = s.key; span.append(k);
      } else {
        span.append(chip(s.key.cell));
      }
      if (s.tail) span.append((/^[,.…]/.test(s.tail) ? "" : " ") + s.tail);
      span.append(" ");
      el.append(span);
    }
    if (t === N) {
      const v = document.createElement("p");
      v.className = "verdict " + p.steps.verdict.cls;
      v.textContent = p.steps.verdict.text;
      el.append(v);
    }
  }

  /* ---------- state ---------- */
  let t = 0, timer = null;
  const tEl = document.getElementById("fig-t");
  const playBtn = document.getElementById("fig-play");
  const scrub = document.getElementById("fig-scrub");
  const dots = [];
  for (let i = 0; i <= N; i++) {
    const b = document.createElement("button");
    b.className = "dot"; b.setAttribute("aria-label", `go to step ${i}`);
    b.addEventListener("click", () => { pause(); setT(i); });
    scrub.appendChild(b); dots.push(b);
  }

  function setT(next) {
    t = Math.max(0, Math.min(N, next));
    tEl.textContent = t;
    dots.forEach((d, i) => d.classList.toggle("on", i <= t));
    for (const p of Object.values(panels)) {
      renderChart(p, t ? p.steps[t - 1] : null);
      renderTrace(p, t);
    }
    // pointed patches persist on the input image
    clearCells("on"); clearCells("flash");
    V1.slice(0, t).forEach((s, i) => {
      if (typeof s.key !== "string") {
        setCell(s.key.cell, "on", true);
        if (i === t - 1) setCell(s.key.cell, "flash", true);
      }
    });
    fig.classList.toggle("done", t === N);
    if (t === N) pause();
  }

  function play() {
    if (timer) return;
    if (t >= N) setT(0);
    playBtn.textContent = "pause"; playBtn.setAttribute("aria-label", "Pause");
    timer = setInterval(() => { if (t < N) setT(t + 1); else pause(); }, STEP_MS);
  }
  function pause() {
    if (!timer) return;
    clearInterval(timer); timer = null;
    playBtn.textContent = "play"; playBtn.setAttribute("aria-label", "Play");
  }

  playBtn.addEventListener("click", () => timer ? pause() : play());
  document.getElementById("fig-step").addEventListener("click", () => { pause(); setT(t + 1); });
  document.getElementById("fig-back").addEventListener("click", () => { pause(); setT(t - 1); });
  document.getElementById("fig-reset").addEventListener("click", () => { pause(); setT(0); });
  document.getElementById("fig-controls").addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { e.preventDefault(); pause(); setT(t + 1); }
    if (e.key === "ArrowLeft")  { e.preventDefault(); pause(); setT(t - 1); }
  });

  Object.values(panels).forEach(buildChart);
  setT(0);

  // Play once when the figure scrolls into view (skipped for reduced motion).
  if (!reduceMotion && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) { io.disconnect(); if (t === 0) play(); }
    }, { threshold: 0.45 });
    io.observe(fig);
  } else if (reduceMotion) {
    setT(N);
  }
})();
