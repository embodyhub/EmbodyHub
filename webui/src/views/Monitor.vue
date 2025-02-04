<template>
  <div class="monitor">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Real-time Performance Monitor</span>
              <el-switch
                v-model="autoRefresh"
                active-text="Auto Refresh"
                inactive-text="Manual Refresh"
              />
            </div>
          </template>
          <div class="chart-container">
            <div class="chart" ref="cpuChart"></div>
            <div class="chart" ref="memoryChart"></div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>GPU Usage</span>
            </div>
          </template>
          <div class="chart" ref="gpuChart"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Network Traffic</span>
            </div>
          </template>
          <div class="chart" ref="networkChart"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'

const autoRefresh = ref(true)
const cpuChart = ref(null)
const memoryChart = ref(null)
const gpuChart = ref(null)
const networkChart = ref(null)

let charts = []
let ws = null

const initChart = (chartRef, title, data = []) => {
  const chart = echarts.init(chartRef)
  const option = {
    title: {
      text: title
    },
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: Array(6).fill(0).map((_, i) => {
        const time = new Date()
        time.setSeconds(time.getSeconds() - (5 - i) * 10)
        return time.toLocaleTimeString('en-US', { hour12: false })
      })
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [{
      data: data.length ? data : Array(6).fill(0),
      type: 'line',
      smooth: true
    }]
  }
  chart.setOption(option)
  return chart
}

const updateCharts = (metrics) => {
  const cpuData = metrics.cpu_usage.reduce((a, b) => a + b, 0) / metrics.cpu_usage.length
  const memData = metrics.memory_usage
  const gpuData = metrics.gpu_info ? metrics.gpu_info.utilization : 0
  const networkData = metrics.network_io.bytes_sent + metrics.network_io.bytes_recv

  charts.forEach((chart, index) => {
    const data = chart.getOption().series[0].data
    data.shift()
    data.push(index === 0 ? cpuData : 
              index === 1 ? memData :
              index === 2 ? gpuData :
              networkData / 1024 / 1024) // Convert to MB for network

    const time = new Date().toLocaleTimeString('en-US', { hour12: false })
    const xAxisData = chart.getOption().xAxis[0].data
    xAxisData.shift()
    xAxisData.push(time)

    chart.setOption({
      xAxis: {
        data: xAxisData
      },
      series: [{
        data: data
      }]
    })
  })
}

const connectWebSocket = () => {
  ws = new WebSocket('ws://localhost:8000/ws/metrics')
  ws.onmessage = (event) => {
    const metrics = JSON.parse(event.data)
    updateCharts(metrics)
  }
  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    autoRefresh.value = false
  }
}

watch(autoRefresh, (newValue) => {
  if (newValue) {
    connectWebSocket()
  } else if (ws) {
    ws.close()
  }
})

onMounted(() => {
  charts = [
    initChart(cpuChart.value, 'CPU Usage'),
    initChart(memoryChart.value, 'Memory Usage'),
    initChart(gpuChart.value, 'GPU Usage'),
    initChart(networkChart.value, 'Network Traffic (MB/s)')
  ]

  if (autoRefresh.value) {
    connectWebSocket()
  }
})

onUnmounted(() => {
  if (ws) {
    ws.close()
  }
  charts.forEach(chart => chart.dispose())
})
</script>

<style scoped>
.monitor {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.chart {
  height: 300px;
}
</style>