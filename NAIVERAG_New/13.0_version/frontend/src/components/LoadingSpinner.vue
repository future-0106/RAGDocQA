<script setup lang="ts">
defineProps<{
  text?: string
}>()
</script>

<template>
  <div class="flex items-center gap-3 text-slate-400 py-4">
    <i class="fas fa-spinner fa-spin text-neon-cyan"></i>
    <span>{{ text || '处理中...' }}</span>
  </div>
</template>
