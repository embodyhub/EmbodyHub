<template>
  <div class="behavior-visualization">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Real-time Behavior Trajectory</span>
              <el-tooltip content="Display agent's behavior decision path">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
          </template>
          <div class="chart" ref="trajectoryChart"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Behavior Prediction Distribution</span>
              <el-tooltip content="Show possible next behaviors and their probabilities">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
          </template>
          <div class="chart" ref="predictionChart"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Behavior Feature Analysis</span>
              <el-radio-group v-model="featureType" size="small">
                <el-radio-button label="temporal">Temporal Features</el-radio-button>
                <el-radio-button label="spatial">Spatial Features</el-radio-button>
                <el-radio-button label="correlation">Correlation Analysis</el-radio-button>
              </el-radio-group>
            </div>
          </template>
          <div class="chart" ref="featureChart"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Behavior Pattern Evolution</span>
              <el-button-group>
                <el-button size="small" @click="exportEvolutionData('png')">Export Image</el-button>
                <el-button size="small" @click="exportEvolutionData('svg')">Export SVG</el-button>
              </el-button-group>
            </div>
          </template>
          <div class="evolution-timeline">
            <div class="chart" ref="evolutionChart"></div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useStore } from 'vuex'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { QuestionFilled } from '@element-plus/icons-vue'

const store = useStore()
const featureType = ref('temporal')

const trajectoryChart = ref(null)
const predictionChart = ref(null)
const featureChart = ref(null)
const evolutionChart = ref(null)

let charts = []

const initTrajectoryChart = () => {
  const chart = echarts.init(trajectoryChart.value)
  const option = {
    tooltip: {
      trigger: 'item'
    },
    series: [{
      type: 'graph',
      layout: 'force',
      data: [],
      links: [],
      categories: [
        { name: 'explore' },
        { name: 'exploit' },
        { name: 'learn' }
      ],
      roam: true,
      label: {
        show: true
      },
      force: {
        repulsion: 100
      }
    }]
  }
  chart.setOption(option)
  return chart
}

const initPredictionChart = () => {
  const chart = echarts.init(predictionChart.value)
  const option = {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c}%'
    },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: '#fff',
        borderWidth: 2
      },
      label: {
        show: false,
        position: 'center'
      },
      emphasis: {
        label: {
          show: true,
          fontSize: '20',
          fontWeight: 'bold'
        }
      },
      labelLine: {
        show: false
      },
      data: []
    }]
  }
  chart.setOption(option)
  return chart
}

const initFeatureChart = () => {
  const chart = echarts.init(featureChart.value)
  const option = {
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: []
    },
    yAxis: {
      type: 'value'
    },
    series: []
  }
  chart.setOption(option)
  return chart
}

const initEvolutionChart = () => {
  const chart = echarts.init(evolutionChart.value)
  const option = {
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'time',
      boundaryGap: false
    },
    yAxis: {
      type: 'value',
      splitLine: {
        show: false
      }
    },
    series: []
  }
  chart.setOption(option)
  return chart
}

const updateCharts = (data) => {
  // 更新行为轨迹图
  const trajectoryData = transformTrajectoryData(data.trajectory)
  charts[0].setOption({
    series: [{
      data: trajectoryData.nodes,
      links: trajectoryData.links
    }]
  })

  // 更新预测分布图
  const predictionData = transformPredictionData(data.predictions)
  charts[1].setOption({
    series: [{
      data: predictionData
    }]
  })

  // 更新特征分析图
  const featureData = transformFeatureData(data.features, featureType.value)
  charts[2].setOption(featureData)

  // 更新演变图
  const evolutionData = transformEvolutionData(data.evolution)
  charts[3].setOption({
    series: evolutionData
  })
}

const transformTrajectoryData = (data) => {
  // 转换轨迹数据为图表所需格式
  return {
    nodes: data.nodes.map(node => ({
      name: node.id,
      value: node.value,
      category: node.type,
      symbolSize: node.value * 20
    })),
    links: data.links.map(link => ({
      source: link.source,
      target: link.target,
      value: link.value
    }))
  }
}

const transformPredictionData = (predictions) => {
  return predictions.map(p => ({
    name: p.pattern,
    value: p.probability * 100
  }))
}

const transformFeatureData = (features, type) => {
  const options = {
    temporal: {
      xAxis: { data: features.timestamps },
      series: features.metrics.map(metric => ({
        name: metric.name,
        type: 'line',
        data: metric.values
      }))
    },
    spatial: {
      series: [{
        type: 'scatter',
        data: features.coordinates
      }]
    },
    correlation: {
      series: [{
        type: 'heatmap',
        data: features.correlationMatrix
      }]
    }
  }
  return options[type]
}

const transformEvolutionData = (evolution) => {
  return evolution.patterns.map(pattern => ({
    name: pattern.name,
    type: 'line',
    smooth: true,
    data: pattern.timeline
  }))
}

const exportEvolutionData = async (format) => {
  try {
    const chart = charts[3]
    const dataUrl = chart.getDataURL({ type: format, pixelRatio: 2 })
    const link = document.createElement('a')
    link.download = `behavior_evolution.${format}`
    link.href = dataUrl
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    ElMessage.success(`成功导出${format.toUpperCase()}格式图表`)
  } catch (error) {
    ElMessage.error('导出失败：' + error.message)
  }
}

// 监听特征类型变化
watch(featureType, () => {
  const data = store.state.behaviorData
  if (data) {
    const featureData = transformFeatureData(data.features, featureType.value)
    charts[2].setOption(featureData)
  }
})

onMounted(() => {
  charts = [
    initTrajectoryChart(),
    initPredictionChart(),
    initFeatureChart(),
    initEvolutionChart()
  ]

  // 初始化数据
  const data = store.state.behaviorData
  if (data) {
    updateCharts(data)
  }

  // 监听窗口大小变化
  window.addEventListener('resize', () => {
    charts.forEach(chart => chart.resize())
  })
})

onUnmounted(() => {
  charts.forEach(chart => {
    chart.dispose()
  })
  window.removeEventListener('resize', () => {
    charts.forEach(chart => chart.resize())
  })
})
</script>

<style scoped>
.behavior-visualization {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart {
  height: 400px;
  width: 100%;
}

.evolution-timeline {
  height: 500px;
  width: 100%;
}
</style>