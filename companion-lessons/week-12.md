# Week 12 Companion - Troubleshooting, Capstone Integration, and Portfolio Defense

[<- Companion index](README.md) · [Master Plan](../MASTER-PLAN.md) · [cert plan](../nvidia-cert-track/month-3-ncp-aio/week-12/plan.md) · [build project](../gpu-engineering-lab/03-scale-and-serve/week-12-capstone/README.md)

## Prerequisite Checklist

- You can triage GPU workload failures by layer: storage, network, driver, runtime, scheduler, app.
- You can run timed lab drills without turning them into open-ended research.
- You can explain the capstone pipeline from data to serving.
- You know which READMEs still lack measured numbers.
- You have a short portfolio tour prepared before exam week begins.

## Mini Lesson

Week 12 is not a learning week in the normal sense. It is a proving week. The right posture is:

1. identify the layer;
2. gather one decisive signal;
3. apply the smallest fix;
4. rerun the check;
5. document the result.

This is also how the portfolio should read: claim, evidence, caveat, next step.

## Math Insight

Troubleshooting performance is often ratio work:

```text
GPU_utilization_low + dataloader_wait_high => input pipeline bottleneck
tokens_per_second_flat + queue_depth_rising => service saturated
multi_gpu_speedup = single_gpu_time / multi_gpu_time
scaling_efficiency = speedup / num_gpus
```

The ratios tell you whether the bottleneck moved after your fix.

## Programming Primer

- Prefer reproducible scripts to manual demos during capstone week.
- Keep commands copy-pasteable and outputs saved.
- For GitHub Pages, every public claim should link to the page or README that proves it.
- The final report should answer: what did you build, what does it prove, what broke, and what would you do with a real 8xH100 box?

## 25-Minute Gate

1. List the five most likely failure layers for a GPU training job.
2. Draft the 90-second repo tour: PROVE, BUILD, SHOW, capstone.
3. Open the root project index and identify every blank headline-result row.
4. Decide what will be cut if exam readiness and capstone polish collide.
