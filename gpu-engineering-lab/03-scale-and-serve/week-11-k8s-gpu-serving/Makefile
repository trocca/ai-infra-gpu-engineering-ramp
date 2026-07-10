# Week 11 — K8s GPU serving stack. All targets run ON the cloud node
# (clone this repo there). The 30-minute recreate test is `make everything`.

KUBECTL ?= kubectl

.PHONY: everything cluster gpu-operator monitoring kai deploy-app timeslicing batch-job \
        status teardown teardown-apps

# ---- the acceptance test: fresh node -> full stack, stopwatch it ----
everything: cluster gpu-operator deploy-app monitoring kai timeslicing batch-job status

cluster:
	bash scripts/00-k3s-install.sh

gpu-operator:
	bash scripts/01-gpu-operator.sh gpu-operator

monitoring:
	bash scripts/01-gpu-operator.sh monitoring

kai:
	bash scripts/01-gpu-operator.sh kai
	$(KUBECTL) apply -f k8s/kai/queue.yaml

deploy-app:
	$(KUBECTL) apply -f k8s/namespace.yaml
	$(KUBECTL) apply -f k8s/deployment.yaml
	$(KUBECTL) apply -f k8s/service.yaml
	$(KUBECTL) rollout status deployment/ferrum-serve -n serving --timeout=600s

timeslicing:
	$(KUBECTL) apply -f k8s/timeslicing-config.yaml
	$(KUBECTL) patch clusterpolicies.nvidia.com/cluster-policy -n gpu-operator \
	  --type merge -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"l4-shared"}}}}'
	@echo "wait ~60s, then verify:  kubectl describe node | grep nvidia.com/gpu"

batch-job:
	$(KUBECTL) apply -f k8s/kai/batch-job.yaml

status:
	$(KUBECTL) get pods -A
	$(KUBECTL) describe node | grep -A2 "nvidia.com/gpu" || true

# ---- teardown (app-level; nuking the whole node also counts) ----
teardown-apps:
	-$(KUBECTL) delete -f k8s/kai/batch-job.yaml
	-$(KUBECTL) delete -f k8s/deployment.yaml -f k8s/service.yaml
	-$(KUBECTL) delete -f k8s/namespace.yaml

teardown: teardown-apps
	-helm uninstall kai-scheduler -n kai-scheduler
	-helm uninstall monitoring -n monitoring
	-helm uninstall gpu-operator -n gpu-operator
	@echo "or simply: /usr/local/bin/k3s-uninstall.sh  (and/or kill the node)"
