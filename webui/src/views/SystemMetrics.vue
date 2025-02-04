<template>
  <div class="metrics-dashboard">
    <div class="metrics-header">
      <h2>System Performance Monitor</h2>
      <div class="refresh-rate">
        <span>Refresh Interval: {{ refreshInterval }} seconds</span>
        <select v-model="refreshInterval">
          <option value="1">1 second</option>
          <option value="5">5 seconds</option>
          <option value="10">10 seconds</option>
        </select>
      </div>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <h3>CPU Usage</h3>
        <div class="chart-container">
          <line-chart :data="cpuData" :options="chartOptions" />
        </div>
      </div>

      <div class="metric-card">
        <h3>Memory Usage</h3>
        <div class="chart-container">
          <line-chart :data="memoryData" :options="chartOptions" />
        </div>
      </div>

      <div class="metric-card">
        <h3>Disk Usage</h3>
        <div class="chart-container">
          <line-chart :data="diskData" :options="chartOptions" />
        </div>
      </div>

      <div class="metric-card" v-if="hasGPU">
        <h3>GPU Usage</h3>
        <div class="chart-container">
          <line-chart :data="gpuData" :options="chartOptions" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { LineChart } from 'vue-chartjs'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

export default {
  name: 'SystemMetrics',
  components: { LineChart },

  setup() {
    const refreshInterval = ref(1)
    const wsConnection = ref(null)
    const hasGPU = ref(false)

    // Initialize chart data
    const initChartData = () => ({
      labels: [],
      datasets: [{
        label: 'Usage (%)',
        data: [],
        borderColor: '#41b883',
        tension: 0.1
      }]
    })

    const cpuData = ref(initChartData())
    const memoryData = ref(initChartData())
    const diskData = ref(initChartData())
    const gpuData = ref(initChartData())

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      }
    }

    const updateMetrics = (metrics) => {
      const timestamp = new Date().toLocaleTimeString()
      const maxDataPoints = 20

      // Update CPU data
      cpuData.value.labels.push(timestamp)
      cpuData.value.datasets[0].data.push(metrics.cpu.percent[0])
      if (cpuData.value.labels.length > maxDataPoints) {
        cpuData.value.labels.shift()
        cpuData.value.datasets[0].data.shift()
      }

      // 更新内存数据
      memoryData.value.labels.push(timestamp)
      memoryData.value.datasets[0].data.push(metrics.memory.virtual.percent)
      if (memoryData.value.labels.length > maxDataPoints) {
        memoryData.value.labels.shift()
        memoryData.value.datasets[0].data.shift()
      }

      // 更新磁盘数据
      diskData.value.labels.push(timestamp)
      diskData.value.datasets[0].data.push(metrics.disk.usage.percent)
      if (diskData.value.labels.length > maxDataPoints) {
        diskData.value.labels.shift()
        diskData.value.datasets[0].data.shift()
      }

      // 更新GPU数据
      if (metrics.gpu) {
        hasGPU.value = true
        gpuData.value.labels.push(timestamp)
        gpuData.value.datasets[0].data.push(
          metrics.gpu.utilization || 
          (metrics.gpu.memory.allocated / metrics.gpu.memory.reserved) * 100
        )
        if (gpuData.value.labels.length > maxDataPoints) {
          gpuData.value.labels.shift()
          gpuData.value.datasets[0].data.shift()
        }
      }
    }

    const connectWebSocket = () => {
      wsConnection.value = new WebSocket('ws://localhost:8000/ws/metrics')
      
      wsConnection.value.onmessage = (event) => {
        const metrics = JSON.parse(event.data)
        updateMetrics(metrics)
      }

      wsConnection.value.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      wsConnection.value.onclose = () => {
        setTimeout(connectWebSocket, 5000) // 重连
      }
    }

    onMounted(() => {
      connectWebSocket()
    })

    onUnmounted(() => {
      if (wsConnection.value) {
        wsConnection.value.close()
      }
    })

    return {
      refreshInterval,
      hasGPU,
      cpuData,
      memoryData,
      diskData,
      gpuData,
      chartOptions
    }
  }
}
</script>

<style scoped>
.metrics-dashboard {
  padding: 20px;
}

.metrics-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.refresh-rate select {
  margin-left: 10px;
  padding: 5px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
}

.metric-card {
  background: white;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.metric-card h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #2c3e50;
}

.chart-container {
  height: 250px;
}
</style>