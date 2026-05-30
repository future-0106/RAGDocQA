<script setup lang="ts">
import TopStatusBar from '@/components/TopStatusBar.vue'
import AppSidebar from '@/components/AppSidebar.vue'
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <TopStatusBar />
    <div class="flex flex-1">
      <AppSidebar />
      <main class="flex-1 p-6 overflow-auto">
        <router-view />
      </main>
    </div>
  </div>
</template>
