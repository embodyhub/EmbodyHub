<template>
  <el-container class="app-container">
    <el-aside width="200px">
      <el-menu
        :router="true"
        class="el-menu-vertical"
        :default-active="$route.path">
        <el-menu-item index="/">
          <el-icon><Monitor /></el-icon>
          <span>Dashboard</span>
        </el-menu-item>
        <el-menu-item index="/models">
          <el-icon><Connection /></el-icon>
          <span>Models</span>
        </el-menu-item>
        <el-menu-item index="/monitor">
          <el-icon><DataLine /></el-icon>
          <span>Performance</span>
        </el-menu-item>
        <el-menu-item index="/behavior">
          <el-icon><TrendCharts /></el-icon>
          <span>Behavior Analysis</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header>
        <div class="header-content">
          <h2>EmbodyHub Management Console</h2>
          <el-tag :type="systemStatusType">{{ systemStatus }}</el-tag>
        </div>
      </el-header>
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useStore } from 'vuex'
import { Monitor, Connection, DataLine } from '@element-plus/icons-vue'

const store = useStore()

const systemStatus = computed(() => {
  const status = store.state.systemStatus
  return status.charAt(0).toUpperCase() + status.slice(1)
})

const systemStatusType = computed(() => {
  const status = store.state.systemStatus
  switch (status) {
    case 'idle': return 'info'
    case 'running': return 'success'
    case 'error': return 'danger'
    default: return 'warning'
  }
})
</script>

<style>
.app-container {
  height: 100vh;
}

.el-header {
  background-color: #fff;
  border-bottom: 1px solid #dcdfe6;
  padding: 0 20px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
}

.el-aside {
  background-color: #304156;
  color: #fff;
}

.el-menu {
  border-right: none;
}

.el-menu-vertical {
  height: 100%;
  background-color: #304156;
}

.el-menu-item {
  color: #fff;
}

.el-menu-item.is-active {
  background-color: #1890ff !important;
}

.el-main {
  background-color: #f0f2f5;
  padding: 20px;
}
</style>