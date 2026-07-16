// Copyright 2025 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package node_info

import (
	resourceapi "k8s.io/api/resource/v1"

	commonconstants "github.com/kai-scheduler/KAI-scheduler/pkg/common/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
)

// A single physical DRA device may be shared by several pods through one
// ResourceClaim with more than one entry in status.reservedFor. Each such pod
// carries the same allocated device in its ResourceClaimInfo, so counting the
// device once per pod (the naive per-task accounting) inflates the node's used
// GPU count above physical capacity and drives IdleVector negative, making the
// whole node unschedulable. draSharedDeviceRefCount tracks how many pods on the
// node currently reference each allocated device (keyed by driver/pool/device)
// so a shared device contributes to UsedVector exactly once.

// draDeviceKey uniquely identifies a physical DRA device on the node.
func draDeviceKey(result resourceapi.DeviceRequestAllocationResult) string {
	return result.Driver + "/" + result.Pool + "/" + result.Device
}

// allocatedGPUDeviceKeys returns the keys of all GPU devices allocated to the
// task via DRA ResourceClaims. Non-GPU devices are ignored: they are not part
// of the GPU accounting that this dedup protects.
func (ni *NodeInfo) allocatedGPUDeviceKeys(task *pod_info.PodInfo) []string {
	var keys []string
	for _, claimAllocation := range task.ResourceClaimInfo {
		if claimAllocation == nil || claimAllocation.Allocation == nil {
			continue
		}
		for _, result := range claimAllocation.Allocation.Devices.Results {
			if !isGPUDRADriver(result.Driver) {
				continue
			}
			keys = append(keys, draDeviceKey(result))
		}
	}
	return keys
}

// isGPUDRADriver reports whether the DRA driver name belongs to an NVIDIA GPU
// device. The device-class name is not present on the allocation result, so the
// driver name is used instead.
func isGPUDRADriver(driver string) bool {
	return driver == commonconstants.NvidiaGpuDraDriver
}

// dedupSharedDRAGpus removes from resourcesToTrack the GPU count that would
// double-count physical DRA devices already referenced by other pods on the
// node. It also updates the node's per-device reference count. It must be
// called once per addTaskResources, before the vector is added to UsedVector.
func (ni *NodeInfo) dedupSharedDRAGpus(task *pod_info.PodInfo, resourcesToTrack resource_info.ResourceVector) {
	alreadyCounted := 0.0
	for _, key := range ni.allocatedGPUDeviceKeys(task) {
		if ni.DRASharedDeviceRefCount[key] > 0 {
			// Another pod on this node already contributed this physical
			// device to the used vector: do not count it again.
			alreadyCounted++
		}
		ni.DRASharedDeviceRefCount[key]++
	}

	if alreadyCounted > 0 {
		current := resourcesToTrack.Get(resource_info.GPUIndex)
		resourcesToTrack.Set(resource_info.GPUIndex, current-alreadyCounted)
	}
}

// releaseSharedDRAGpus is the inverse of dedupSharedDRAGpus: it decrements the
// per-device reference count and adds back the GPU count for devices that
// remain referenced by other pods (and were therefore never subtracted on this
// task's removal path). It must be called once per removeTaskResources.
func (ni *NodeInfo) releaseSharedDRAGpus(task *pod_info.PodInfo, resourcesToTrack resource_info.ResourceVector) {
	stillShared := 0.0
	for _, key := range ni.allocatedGPUDeviceKeys(task) {
		if ni.DRASharedDeviceRefCount[key] > 1 {
			// The device stays referenced by another pod after this removal:
			// it must remain in the used vector, so this task's removal must
			// not subtract it.
			stillShared++
		}
		if ni.DRASharedDeviceRefCount[key] > 0 {
			ni.DRASharedDeviceRefCount[key]--
		}
		if ni.DRASharedDeviceRefCount[key] == 0 {
			delete(ni.DRASharedDeviceRefCount, key)
		}
	}

	if stillShared > 0 {
		current := resourcesToTrack.Get(resource_info.GPUIndex)
		resourcesToTrack.Set(resource_info.GPUIndex, current-stillShared)
	}
}
