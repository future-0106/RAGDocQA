<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { getStatus } from '@/api/status'

const currentModel = ref('Loading...')
const currentTime = ref('')

function updateTime() {
  currentTime.value = new Date().toLocaleTimeString()
}

async function loadModel() {
  try {
    const res = await getStatus()
    if (res.success && res.data) {
      currentModel.value = res.data.models.llm
    }
  } catch {
    currentModel.value = 'Unavailable'
  }
}

let timer: ReturnType<typeof setInterval>
onMounted(() => {
  loadModel()
  updateTime()
  timer = setInterval(updateTime, 1000)
})
onUnmounted(() => clearInterval(timer))
</script>

<template>
  <header class="top-bar">
    <div class="top-bar-left">
      <img src="/images/下载.jpg" alt="logo" class="logo-img">
      <span class="font-semibold text-white text-sm">RAG 智能问答系统</span>
    </div>
    <div class="top-bar-right">
      <span class="flex items-center gap-1.5 text-xs text-slate-400">
        <span class="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
        运行中
      </span>
      <span class="text-xs text-slate-500 hidden sm:inline">Model: {{ currentModel }}</span>
      <span class="text-xs text-slate-500">{{ currentTime }}</span>
    </div>
  </header>
</template>

<style scoped>
.top-bar {
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  background: rgba(15, 15, 24, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid #1e1e2e;
  position: sticky;
  top: 0;
  z-index: 50;
}

.top-bar-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.top-bar-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.text-neon-cyan {
  color: #00f5ff;
}

.logo-img {
  width: 28px;
  height: 28px;
  object-fit: contain;
}
</style>
