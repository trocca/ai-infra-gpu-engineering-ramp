(function () {
  "use strict";

  const COOKIE_NAME = "aiInfraRampProgressV1";
  const PANEL_COOKIE = "aiInfraRampProgressPanel";
  const COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

  const coreWeeks = [
    {
      id: "w01",
      label: "Week 1 - AI basics + autograd",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-01.md"],
        ["s", "Study NCA AI basics", "nvidia-cert-track/month-1-nca-aiio/week-1/plan.md"],
        ["b", "Build autograd + Rust ramp", "gpu-engineering-lab/01-foundations/week-01-autograd-from-scratch/README.md"],
        ["g", "Friday self-check + progress row", "nvidia-cert-track/month-1-nca-aiio/week-1/self-check.md"]
      ]
    },
    {
      id: "w02",
      label: "Week 2 - GPU hardware + CUDA",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-02.md"],
        ["s", "Study GPU hardware", "nvidia-cert-track/month-1-nca-aiio/week-2/plan.md"],
        ["b", "Build CUDA-in-Rust basics", "gpu-engineering-lab/01-foundations/week-02-cuda-basics/README.md"],
        ["g", "Friday self-check + progress row", "nvidia-cert-track/month-1-nca-aiio/week-2/self-check.md"]
      ]
    },
    {
      id: "w03",
      label: "Week 3 - Networking + SGEMM",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-03.md"],
        ["s", "Study networking and datacenter design", "nvidia-cert-track/month-1-nca-aiio/week-3/plan.md"],
        ["b", "Build SGEMM optimization ladder", "gpu-engineering-lab/01-foundations/week-03-matmul-optimization/README.md"],
        ["g", "Friday self-check + progress row", "nvidia-cert-track/month-1-nca-aiio/week-3/self-check.md"]
      ]
    },
    {
      id: "w04",
      label: "Week 4 - AI operations + exam",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-04.md"],
        ["s", "Study NCA operations and review", "nvidia-cert-track/month-1-nca-aiio/week-4/plan.md"],
        ["b", "Build PyTorch custom ops", "gpu-engineering-lab/01-foundations/week-04-pytorch-custom-ops/README.md"],
        ["g", "Mock exam, self-check, NCA exam", "nvidia-cert-track/month-1-nca-aiio/mock-exam.md"]
      ]
    },
    {
      id: "w05",
      label: "Week 5 - Transformers + GPT",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-05.md"],
        ["s", "Study LLM architecture", "nvidia-cert-track/month-2-ncp-genl/week-5/plan.md"],
        ["b", "Build GPT from scratch", "gpu-engineering-lab/02-llm-engineering/week-05-gpt-from-scratch/README.md"],
        ["g", "Friday self-check + results", "nvidia-cert-track/month-2-ncp-genl/week-5/self-check.md"]
      ]
    },
    {
      id: "w06",
      label: "Week 6 - Fine-tuning + LoRA",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-06.md"],
        ["s", "Study PEFT and evaluation", "nvidia-cert-track/month-2-ncp-genl/week-6/plan.md"],
        ["b", "Build LoRA from scratch", "gpu-engineering-lab/02-llm-engineering/week-06-lora-from-scratch/README.md"],
        ["g", "Friday self-check + results", "nvidia-cert-track/month-2-ncp-genl/week-6/self-check.md"]
      ]
    },
    {
      id: "w07",
      label: "Week 7 - Optimization + Triton",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-07.md"],
        ["s", "Study optimization and GPU acceleration", "nvidia-cert-track/month-2-ncp-genl/week-7/plan.md"],
        ["b", "Build Triton kernels + quantization", "gpu-engineering-lab/02-llm-engineering/week-07-triton-quantization/README.md"],
        ["g", "Friday self-check + results", "nvidia-cert-track/month-2-ncp-genl/week-7/self-check.md"]
      ]
    },
    {
      id: "w08",
      label: "Week 8 - Deployment + exam",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-08.md"],
        ["s", "Study deployment and review", "nvidia-cert-track/month-2-ncp-genl/week-8/plan.md"],
        ["b", "Build ferrum-serve", "gpu-engineering-lab/02-llm-engineering/week-08-mini-inference-server/README.md"],
        ["g", "Mock exam, self-check, GENL exam", "nvidia-cert-track/month-2-ncp-genl/mock-exam.md"]
      ]
    },
    {
      id: "w09",
      label: "Week 9 - Cluster install + DDP",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-09.md"],
        ["s", "Study BCM, Slurm, K8s setup", "nvidia-cert-track/month-3-ncp-aio/week-9/plan.md"],
        ["b", "Build distributed training internals", "gpu-engineering-lab/03-scale-and-serve/week-09-distributed-training/README.md"],
        ["g", "Friday self-check + results", "nvidia-cert-track/month-3-ncp-aio/week-9/self-check.md"]
      ]
    },
    {
      id: "w10",
      label: "Week 10 - Admin + parallelism",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-10.md"],
        ["s", "Study admin, Run:ai, MIG", "nvidia-cert-track/month-3-ncp-aio/week-10/plan.md"],
        ["b", "Build tensor + pipeline parallelism", "gpu-engineering-lab/03-scale-and-serve/week-10-parallelism-internals/README.md"],
        ["g", "Friday self-check + results", "nvidia-cert-track/month-3-ncp-aio/week-10/self-check.md"]
      ]
    },
    {
      id: "w11",
      label: "Week 11 - Workloads + K8s serving",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-11.md"],
        ["s", "Study workload management", "nvidia-cert-track/month-3-ncp-aio/week-11/plan.md"],
        ["b", "Build K8s GPU serving stack", "gpu-engineering-lab/03-scale-and-serve/week-11-k8s-gpu-serving/README.md"],
        ["g", "Friday self-check + results", "nvidia-cert-track/month-3-ncp-aio/week-11/self-check.md"]
      ]
    },
    {
      id: "w12",
      label: "Week 12 - Troubleshooting + capstone",
      items: [
        ["p", "Prep companion lesson", "companion-lessons/week-12.md"],
        ["s", "Study AIO lab drills and review", "nvidia-cert-track/month-3-ncp-aio/week-12/plan.md"],
        ["b", "Build capstone pipeline", "gpu-engineering-lab/03-scale-and-serve/week-12-capstone/README.md"],
        ["g", "Mock exam, capstone defense, AIO exam", "nvidia-cert-track/month-3-ncp-aio/mock-exam.md"]
      ]
    }
  ];

  const supportGroups = [
    {
      id: "cuda",
      label: "Optional C++/CUDA skill gym",
      items: [
        ["01", "01 execution model", "cpp-cuda-track/01-execution-model/README.md"],
        ["02", "02 memory hierarchy", "cpp-cuda-track/02-memory-hierarchy/README.md"],
        ["03", "03 SAXPY", "cpp-cuda-track/03-data-parallel-saxpy/README.md"],
        ["04", "04 reduction", "cpp-cuda-track/04-reduction/README.md"],
        ["05", "05 scan and histogram", "cpp-cuda-track/05-scan-histogram/README.md"],
        ["06", "06 matmul tiling", "cpp-cuda-track/06-matmul-tiling/README.md"],
        ["07", "07 async overlap", "cpp-cuda-track/07-async-overlap/README.md"],
        ["08", "08 atomics and memory model", "cpp-cuda-track/08-sync-atomics-memory-model/README.md"],
        ["09", "09 profiling roofline", "cpp-cuda-track/09-profiling-roofline/README.md"],
        ["10", "10 advanced GPU", "cpp-cuda-track/10-advanced-gpu/README.md"],
        ["11", "11 multi-device", "cpp-cuda-track/11-multi-device/README.md"],
        ["12", "12 PyTorch extension", "cpp-cuda-track/12-capstone-pytorch-extension/README.md"]
      ]
    },
    {
      id: "hf",
      label: "Just-in-time HF playbook references",
      items: [
        ["05", "Week 5 memory ledger", "references/hf-ultrascale-playbook.md#week-5---transformer-memory-and-single-gpu-training"],
        ["06", "Week 6 global batch and ZeRO", "references/hf-ultrascale-playbook.md#week-6---fine-tuning-memory-pressure-and-global-batch-math"],
        ["07", "Week 7 kernels and precision", "references/hf-ultrascale-playbook.md#week-7---kernels-flashattention-and-mixed-precision"],
        ["08", "Week 8 serving carryover", "references/hf-ultrascale-playbook.md#week-8---serving-memory-and-precision-carryover"],
        ["09", "Week 9 DP, ZeRO, collectives", "references/hf-ultrascale-playbook.md#week-9---data-parallelism-zero-collectives-and-profiling"],
        ["10", "Week 10 5D strategy", "references/hf-ultrascale-playbook.md#week-10---model-parallelism-and-5d-strategy"],
        ["11", "Week 11 observability", "references/hf-ultrascale-playbook.md#week-11---benchmarking-observability-and-cluster-reality"],
        ["12", "Week 12 scale-up story", "references/hf-ultrascale-playbook.md#week-12---capstone-defense-and-scale-up-story"]
      ]
    }
  ];

  let completed = new Set(readCookieSet(COOKIE_NAME));

  function readCookie(name) {
    const prefix = name + "=";
    return document.cookie
      .split(";")
      .map((part) => part.trim())
      .find((part) => part.indexOf(prefix) === 0)
      ?.slice(prefix.length) || "";
  }

  function writeCookie(name, value) {
    document.cookie = [
      name + "=" + encodeURIComponent(value),
      "Max-Age=" + COOKIE_MAX_AGE,
      "Path=/",
      "SameSite=Lax"
    ].join("; ");
  }

  function readCookieSet(name) {
    const raw = decodeURIComponent(readCookie(name));
    return raw ? raw.split(".").filter(Boolean) : [];
  }

  function persist() {
    writeCookie(COOKIE_NAME, Array.from(completed).sort().join("."));
  }

  function cookieWorks() {
    const probe = "aiInfraRampCookieProbe";
    writeCookie(probe, "1");
    const ok = readCookie(probe) === "1";
    document.cookie = probe + "=; Max-Age=0; Path=/; SameSite=Lax";
    return ok;
  }

  function urlFor(path) {
    const root = getRootPath();
    return root + path.replace(/^\//, "");
  }

  function getRootPath() {
    const script = document.querySelector('script[src*="assets/progress.js"]');
    if (!script) return "/";
    const src = script.getAttribute("src") || "";
    return src.replace(/assets\/progress\.js(?:\?.*)?$/, "");
  }

  function itemId(groupId, itemKey) {
    return groupId + "-" + itemKey;
  }

  function buildItem(groupId, item) {
    const id = itemId(groupId, item[0]);
    const row = document.createElement("div");
    row.className = "ramp-progress-item";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = "ramp-" + id;
    input.checked = completed.has(id);
    input.dataset.progressId = id;

    const label = document.createElement("label");
    label.setAttribute("for", input.id);

    const labelText = document.createElement("span");
    labelText.className = "ramp-progress-item__label";
    labelText.textContent = item[1];

    const link = document.createElement("a");
    link.className = "ramp-progress-item__link";
    link.href = urlFor(item[2]);
    link.textContent = "Open step";
    link.addEventListener("click", (event) => event.stopPropagation());

    label.append(labelText, link);
    row.append(input, label);
    return row;
  }

  function buildSection(group, openByDefault) {
    const details = document.createElement("details");
    details.className = "ramp-progress-section";
    details.open = openByDefault;

    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.textContent = group.label;

    const meta = document.createElement("span");
    meta.className = "ramp-progress-section__meta";
    meta.dataset.groupMeta = group.id;

    summary.append(title, meta);

    const items = document.createElement("div");
    items.className = "ramp-progress-items";
    group.items.forEach((item) => items.append(buildItem(group.id, item)));

    details.append(summary, items);
    return details;
  }

  function createPanel() {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "ramp-progress-toggle";
    toggle.setAttribute("aria-controls", "ramp-progress-panel");
    toggle.setAttribute("aria-expanded", "false");
    toggle.innerHTML = '<span class="ramp-progress-toggle__meter" aria-hidden="true"></span><span class="ramp-progress-toggle__text">Progress</span>';

    const panel = document.createElement("aside");
    panel.className = "ramp-progress-panel";
    panel.id = "ramp-progress-panel";
    panel.setAttribute("aria-label", "Study progress");

    const header = document.createElement("header");
    header.className = "ramp-progress-header";
    header.innerHTML = [
      '<div class="ramp-progress-titlebar">',
      '<div><h2 class="ramp-progress-title">Study Progress</h2>',
      '<p class="ramp-progress-subtitle">Saved in this browser with a cookie. No account or database.</p></div>',
      '<button class="ramp-progress-close" type="button" aria-label="Close progress panel">x</button>',
      '</div>',
      '<div class="ramp-progress-summary">',
      '<div class="ramp-progress-ring"><div class="ramp-progress-ring__inner" data-progress-percent>0%</div></div>',
      '<div><strong>Core path</strong><div class="ramp-progress-meter"><div class="ramp-progress-meter__bar"></div></div>',
      '<p class="ramp-progress-count" data-progress-count>0 of 48 core steps complete</p></div>',
      '</div>'
    ].join("");

    const body = document.createElement("div");
    body.className = "ramp-progress-body";

    const coreGroup = {
      id: "core",
      label: "Core 12-week path",
      items: coreWeeks.flatMap((week) => week.items.map((item) => [week.id + item[0], week.label + ": " + item[1], item[2]]))
    };
    body.append(buildSection(coreGroup, true));
    supportGroups.forEach((group) => body.append(buildSection(group, false)));

    const footer = document.createElement("footer");
    footer.className = "ramp-progress-footer";
    footer.innerHTML = [
      '<span>Tip: optional lanes do not count toward the core percentage.</span>',
      '<span class="ramp-progress-cookie-warning">Cookies appear to be blocked, so progress may not persist after this page closes.</span>',
      '<button class="ramp-progress-reset" type="button">Reset progress</button>'
    ].join("");

    panel.append(header, body, footer);
    document.body.append(toggle, panel);

    return { toggle, panel };
  }

  function setPanelOpen(open, controls) {
    document.body.classList.toggle("ramp-progress-open", open);
    controls.toggle.setAttribute("aria-expanded", String(open));
    writeCookie(PANEL_COOKIE, open ? "open" : "closed");
  }

  function updateProgress() {
    const coreIds = coreWeeks.flatMap((week) => week.items.map((item) => itemId("core", week.id + item[0])));
    const done = coreIds.filter((id) => completed.has(id)).length;
    const total = coreIds.length;
    const percent = total ? Math.round((done / total) * 100) : 0;

    document.documentElement.style.setProperty("--ramp-progress", percent + "%");
    document.querySelectorAll("[data-progress-percent]").forEach((node) => {
      node.textContent = percent + "%";
    });
    document.querySelectorAll("[data-progress-count]").forEach((node) => {
      node.textContent = done + " of " + total + " core steps complete";
    });

    const groupIds = ["core"].concat(supportGroups.map((group) => group.id));
    groupIds.forEach((groupId) => {
      const inputs = Array.from(document.querySelectorAll('input[data-progress-id^="' + groupId + '-"]'));
      const checked = inputs.filter((input) => input.checked).length;
      const meta = document.querySelector('[data-group-meta="' + groupId + '"]');
      if (meta) meta.textContent = checked + "/" + inputs.length;
    });
  }

  function bindEvents(controls) {
    controls.toggle.addEventListener("click", () => {
      setPanelOpen(!document.body.classList.contains("ramp-progress-open"), controls);
    });

    controls.panel.querySelector(".ramp-progress-close").addEventListener("click", () => {
      setPanelOpen(false, controls);
    });

    controls.panel.querySelector(".ramp-progress-reset").addEventListener("click", () => {
      if (!window.confirm("Reset all progress checkmarks for this browser?")) return;
      completed = new Set();
      persist();
      document.querySelectorAll("input[data-progress-id]").forEach((input) => {
        input.checked = false;
      });
      updateProgress();
    });

    controls.panel.addEventListener("change", (event) => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement) || !input.dataset.progressId) return;
      if (input.checked) completed.add(input.dataset.progressId);
      else completed.delete(input.dataset.progressId);
      persist();
      updateProgress();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.body.classList.contains("ramp-progress-open")) {
        setPanelOpen(false, controls);
      }
    });
  }

  function init() {
    if (!document.body || document.querySelector(".ramp-progress-panel")) return;
    const controls = createPanel();
    bindEvents(controls);
    updateProgress();

    if (!cookieWorks()) {
      controls.panel.querySelector(".ramp-progress-cookie-warning")?.classList.add("is-visible");
    }

    setPanelOpen(readCookie(PANEL_COOKIE) === "open", controls);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
