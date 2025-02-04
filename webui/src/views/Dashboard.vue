<template>
  <div class="dashboard">
    <el-row :gutter="20">
      <el-col :span="8">
        <el-card class="status-card">
          <template #header>
            <div class="card-header">
              <span>System Status</span>
            </div>
          </template>
          <el-descriptions :column="1">
            <el-descriptions-item label="Running Status">
              <el-tag :type="systemStatusType">{{ systemStatus }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="Loaded Models">
              {{ modelCount }}
            </el-descriptions-item>
            <el-descriptions-item label="Uptime">
              {{ uptime }}
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
      <el-col :span="16">
        <el-card class="chart-card">
          <template #header>
            <div class="card-header">
              <span>Performance Monitor</span>
            </div>
          </template>
          <div class="chart" ref="performanceChart"></div>
        </el-card>
      </el-col>
    </el-row>
    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="24">
        <el-card class="recent-card">
          <template #header>
            <div class="card-header">
              <span>Recent Activities</span>
            </div>
          </template>
          <el-timeline>
            <el-timeline-item
              v-for="(activity, index) in recentActivities"
              :key="index"
              :timestamp="activity.time"
              :type="activity.type">
              {{ activity.content }}
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useStore } from 'vuex'
import * as echarts from 'echarts'

const store = useStore()
const performanceChart = ref(null)
let chart = null

const systemStatus = computed(() => store.state.systemStatus)
const systemStatusType = computed(() => {
  switch (store.state.systemStatus) {
    case 'idle': return 'info'
    case 'running': return 'success'
    case 'error': return 'danger'
    default: return 'warning'
  }
})

const modelCount = computed(() => store.state.models.length)
const uptime = ref('2 hours 30 minutes')

const recentActivities = ref([
  {
    content: 'New model loaded: PolicyNetwork',
    time: '10 minutes ago',
    type: 'success'
  },
  {
    content: 'System status updated: Running',
    time: '30 minutes ago',
    type: 'info'
  },
  {
    content: 'Performance optimization completed',
    time: '1 hour ago',
    type: 'success'
  }
])

onMounted(() => {
  chart = echarts.init(performanceChart.value)
  const option = {
    title: {
      text: 'System Resource Usage'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['CPU Usage', 'Memory Usage']
    },
    xAxis: {
      type: 'category',
      data: ['12:00', '12:05', '12:10', '12:15', '12:20'],
      name: 'Time'
    },
    yAxis: {
      type: 'value',
      name: 'Usage',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [
      {
        name: 'CPU Usage',
        type: 'line',
        data: [30, 35, 33, 38, 32]
      },
      {
        name: 'Memory Usage',
        type: 'line',
        data: [45, 48, 46, 50, 47]
      }
    ]
  }
  chart.setOption(option)
})
</script>

<style scoped>
.dashboard {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart {
  height: 300px;
}

.el-card {
  margin-bottom: 20px;
}

.status-card :deep(.el-descriptions) {
  margin-top: 20px;
}

.chart-card {
  height: 400px;
}
</style>