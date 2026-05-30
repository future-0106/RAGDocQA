<script setup lang="ts">
defineProps<{
  icon: string
  value: number | string
  label: string
  color: 'cyan' | 'purple' | 'green' | 'orange'
  iconColor?: string
}>()
</script>

<template>
  <div :class="['stat-card', color]">
    <i :class="['fas', icon, 'text-3xl mb-2']" :style="{ color: iconColor || '#00f5ff' }"></i>
    <div class="text-3xl font-bold text-white">{{ value }}</div>
    <div class="text-sm text-slate-400">{{ label }}</div>
  </div>
</template>

<style scoped>
.cyan {
  background: linear-gradient(135deg, #0a3d4a 0%, #0a2d3d 100%);
  border: 1px solid rgba(0, 245, 255, 0.3);
}
.purple {
  background: linear-gradient(135deg, #1e1a3a 0%, #2a1a4a 100%);
  border: 1px solid rgba(139, 92, 246, 0.3);
}
.green {
  background: linear-gradient(135deg, #0a2d1a 0%, #0a3d24 100%);
  border: 1px solid rgba(16, 185, 129, 0.3);
}
.orange {
  background: linear-gradient(135deg, #3d2a0a 0%, #3d1a0a 100%);
  border: 1px solid rgba(245, 158, 11, 0.3);
}
</style>
