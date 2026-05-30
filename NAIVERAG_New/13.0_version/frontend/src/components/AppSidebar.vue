<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()

const navItems = [
  { route: '/chat', icon: 'fa-comments', label: '智能问答' },
  { route: '/upload', icon: 'fa-cloud-upload-alt', label: '文件上传' },
  { route: '/files', icon: 'fa-folder-open', label: '文件管理' },
  { route: '/models', icon: 'fa-microchip', label: '模型管理' },
  { route: '/status', icon: 'fa-chart-line', label: '系统状态' },
  { route: '/history', icon: 'fa-history', label: '查询历史' },
]

function navigateTo(path: string) {
  router.push(path)
}
</script>

<template>
  <aside class="sidebar">
    <nav class="flex flex-col items-center gap-1 pt-3">
      <div
        v-for="item in navItems"
        :key="item.route"
        class="sidebar-item group"
        :class="{ active: route.path === item.route }"
        @click="navigateTo(item.route)"
      >
        <i :class="['fas', item.icon, 'text-lg']"></i>
        <div class="sidebar-tooltip">{{ item.label }}</div>
      </div>
    </nav>
    <div class="mt-auto pb-4 flex justify-center">
      <span class="w-2 h-2 bg-green-500 rounded-full" title="系统运行中"></span>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 72px;
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, #0f0f18 0%, #161622 100%);
  border-right: 1px solid #1e1e2e;
  min-height: calc(100vh - 48px);
  position: relative;
  flex-shrink: 0;
}

.sidebar-item {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #475569;
  cursor: pointer;
  border-radius: 12px;
  transition: all 0.2s ease;
  position: relative;
}

.sidebar-item:hover {
  background: rgba(0, 245, 255, 0.08);
  color: #00f5ff;
}

.sidebar-item.active {
  background: rgba(0, 245, 255, 0.12);
  color: #00f5ff;
  box-shadow: inset 3px 0 0 #00f5ff;
  border-radius: 12px 0 0 12px;
}

.sidebar-tooltip {
  position: absolute;
  left: calc(100% + 12px);
  top: 50%;
  transform: translateY(-50%);
  background: #1e1e2e;
  color: #e2e8f0;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  white-space: nowrap;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
  border: 1px solid #2a2a3e;
  z-index: 100;
}

.sidebar-item:hover .sidebar-tooltip {
  opacity: 1;
}
</style>
