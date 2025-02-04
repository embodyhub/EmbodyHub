<template>
  <div class="behavior-analysis">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="overview-card">
          <template #header>
            <div class="card-header">
              <h3>Agent Behavior Overview</h3>
              <el-button-group>
                <el-button type="primary" @click="refreshData" :loading="loading.overview">
                  <el-icon><Refresh /></el-icon> Refresh
                </el-button>
                <el-button @click="exportReport" :loading="loading.export">
                  <el-icon><Download /></el-icon> Export Report
                </el-button>
              </el-button-group>
            </div>
          </template>
          <el-row :gutter="20">
            <el-col :span="6" v-for="metric in overviewMetrics" :key="metric.label">
              <div class="metric-card">
                <h4>{{ metric.label }}</h4>
                <p class="metric-value">{{ metric.value }}</p>
                <p class="metric-trend" :class="metric.trend > 0 ? 'positive' : 'negative'">
                  {{ metric.trend > 0 ? '↑' : '↓' }} {{ Math.abs(metric.trend).toFixed(2) }}%
                </p>
              </div>
            </el-col>
          </el-row>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="chart-row">
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Behavior Pattern Distribution</span>
              <el-select v-model="timeRange" size="small" @change="updateCharts">
                <el-option label="Last 1 Hour" value="hour" />
                <el-option label="Last 24 Hours" value="day" />
                <el-option label="Last 7 Days" value="week" />
              </el-select>
            </div>
          </template>
          <div class="chart" ref="patternChart"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Prediction Accuracy Trend</span>
              <el-tooltip content="Shows the trend of prediction model accuracy">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
          </template>
          <div class="chart" ref="accuracyChart"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Performance Metrics Analysis</span>
              <div class="header-actions">
                <el-date-picker
                  v-model="dateRange"
                  type="daterange"
                  range-separator="to"
                  start-placeholder="Start Date"
                  end-placeholder="End Date"
                  size="small"
                  @change="updatePerformanceData"
                />
                <el-button-group>
                  <el-button size="small" type="primary" @click="exportReport('json')">
                    <el-icon><Download /></el-icon> JSON
                  </el-button>
                  <el-button size="small" type="success" @click="exportReport('csv')">
                    <el-icon><Document /></el-icon> CSV
                  </el-button>
                </el-button-group>
              </div>
            </div>
          </template>
          <el-table :data="performanceData" style="width: 100%" v-loading="loading.performance">
            <el-table-column prop="timestamp" label="Time" width="180" />
            <el-table-column prop="avgReward" label="Average Reward">
              <template #default="{ row }">
                <el-progress
                  :percentage="(row.avgReward + 1) * 50"
                  :color="getRewardColor(row.avgReward)"
                />
              </template>
            </el-table-column>
            <el-table-column prop="responseTime" label="响应时间(ms)">
              <template #default="{ row }">
                {{ row.responseTime.toFixed(2) }}
              </template>
            </el-table-column>
            <el-table-column prop="confidence" label="预测置信度">
              <template #default="{ row }">
                <el-progress
                  :percentage="row.confidence * 100"
                  :status="getConfidenceStatus(row.confidence)"
                />
              </template>
            </el-table-column>
            <el-table-column prop="patternType" label="主要行为模式">
              <template #default="{ row }">
                <el-tag :type="getPatternType(row.patternType)">
                  {{ row.patternType }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>优化建议</span>
              <el-button size="small" type="primary" @click="refreshRecommendations">
                <el-icon><Refresh /></el-icon> Refresh
              </el-button>
            </div>
          </template>
          <el-timeline v-loading="loading.recommendations">
            <el-timeline-item
              v-for="(rec, index) in recommendations"
              :key="index"
              :type="getSeverityType(rec.severity)"
              :timestamp="rec.timestamp"
              :hollow="rec.severity === 'info'"
            >
              <h4>{{ rec.title }}</h4>
              <p>{{ rec.message }}</p>
              <el-button
                v-if="rec.action"
                size="small"
                :type="getActionType(rec.severity)"
                @click="handleRecommendationAction(rec)"
              >
                {{ rec.action }}
              </el-button>
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useStore } from 'vuex'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { QuestionFilled, Download, Document, Refresh } from '@element-plus/icons-vue'
import {
  getBehaviorAnalysis,
  getPerformanceMetrics,
  exportReport as exportReportAPI,
  getRecommendations as getRecommendationsAPI
} from '../api/behavior'

const store = useStore()
const patternChart = ref(null)
const accuracyChart = ref(null)
const timeRange = ref('hour')
const dateRange = ref([])
const performanceData = ref([])
const recommendations = ref([])
const loading = ref({
  overview: false,
  performance: false,
  export: false,
  recommendations: false
})

const overviewMetrics = ref([
  { label: 'Total Behaviors', value: 0, trend: 0 },
  { label: 'Average Response Time', value: '0ms', trend: 0 },
  { label: 'Prediction Accuracy', value: '0%', trend: 0 },
  { label: 'Pattern Stability', value: '0%', trend: 0 }
])

// 自动刷新定时器
let refreshTimer = null

// 自动刷新数据
const startAutoRefresh = () => {
  refreshTimer = setInterval(() => {
    refreshData()
  }, 30000) // 每30秒刷新一次
}

// 停止自动刷新
const stopAutoRefresh = () => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
}

let charts = {
  pattern: null,
  accuracy: null
}

// 初始化图表
const initCharts = () => {
  charts.pattern = echarts.init(patternChart.value)
  charts.accuracy = echarts.init(accuracyChart.value)
  
  window.addEventListener('resize', () => {
    charts.pattern?.resize()
    charts.accuracy?.resize()
  })
}

// 更新图表数据
const updateCharts = async () => {
  try {
    const data = await getBehaviorAnalysis(timeRange.value)
    
    // Update pattern distribution chart
    charts.pattern.setOption({
      title: { text: 'Behavior Pattern Distribution' },
      tooltip: { trigger: 'item' },
      legend: { orient: 'vertical', left: 'left' },
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
        labelLine: { show: false },
        data: data.patterns
      }]
    })
    
    // Update accuracy trend chart
    charts.accuracy.setOption({
      title: { text: 'Prediction Accuracy Trend' },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'time' },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: [{
        name: 'Accuracy',
        type: 'line',
        smooth: true,
        data: data.accuracyTrend,
        areaStyle: {
          opacity: 0.3
        }
      }]
    })
  } catch (error) {
    ElMessage.error('Failed to update chart data')
  }
}

// 更新性能数据
const updatePerformanceData = async () => {
  if (!dateRange.value?.length) return
  
  loading.value.performance = true
  try {
    const [startTime, endTime] = dateRange.value
    const data = await getPerformanceMetrics(startTime, endTime)
    performanceData.value = data
  } catch (error) {
    ElMessage.error('获取性能数据失败')
  } finally {
    loading.value.performance = false
  }
}

// 导出报告
const exportReport = async (format) => {
  try {
    await exportReportAPI(format)
    ElMessage.success(`成功导出${format.toUpperCase()}报告`)
  } catch (error) {
    ElMessage.error('导出报告失败')
  }
}

// 刷新优化建议
const refreshRecommendations = async () => {
  loading.value.recommendations = true
  try {
    recommendations.value = await getRecommendationsAPI()
  } catch (error) {
    ElMessage.error('获取优化建议失败')
  } finally {
    loading.value.recommendations = false
  }
}

// 处理优化建议动作
const handleRecommendationAction = (recommendation) => {
  // 实现具体的优化建议处理逻辑
  ElMessage.info(`执行优化建议: ${recommendation.title}`)
}

// 工具函数
const getRewardColor = (reward) => {
  if (reward > 0.5) return '#67C23A'
  if (reward > 0) return '#E6A23C'
  return '#F56C6C'
}

const getConfidenceStatus = (confidence) => {
  if (confidence >= 0.8) return 'success'
  if (confidence >= 0.6) return 'warning'
  return 'exception'
}

const getPatternType = (pattern) => {
  const types = {
    explore: 'info',
    exploit: 'success',
    learn: 'warning'
  }
  return types[pattern] || 'info'
}

const getSeverityType = (severity) => {
  const types = {
    critical: 'danger',
    warning: 'warning',
    info: 'info'
  }
  return types[severity] || 'info'
}

const getActionType = (severity) => {
  const types = {
    critical: 'danger',
    warning: 'warning',
    info: 'primary'
  }
  return types[severity] || 'primary'
}

// 刷新数据
const refreshData = async () => {
  loading.value.overview = true
  try {
    const data = await getBehaviorAnalysis()
    updateOverviewMetrics(data)
    updateCharts()
    refreshRecommendations()
  } catch (error) {
    ElMessage.error('刷新数据失败')
  } finally {
    loading.value.overview = false
  }
}

// 更新概览指标
const updateOverviewMetrics = (data) => {
  if (!data) return
  
  overviewMetrics.value = [
    {
      label: '行为总数',
      value: data.totalBehaviors || 0,
      trend: data.behaviorsTrend || 0
    },
    {
      label: '平均响应时间',
      value: `${(data.avgResponseTime || 0).toFixed(2)}ms`,
      trend: data.responseTimeTrend || 0
    },
    {
      label: '预测准确率',
      value: `${((data.predictionAccuracy || 0) * 100).toFixed(1)}%`,
      trend: data.accuracyTrend || 0
    },
    {
      label: '模式稳定性',
      value: `${((data.patternStability || 0) * 100).toFixed(1)}%`,
      trend: data.stabilityTrend || 0
    }
  ]
}

// 生命周期钩子
onMounted(() => {
  initCharts()
  refreshData()
  startAutoRefresh()
})

onUnmounted(() => {
  charts.pattern?.dispose()
  charts.accuracy?.dispose()
  stopAutoRefresh()
})
</script>

<style scoped>
.behavior-analysis {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}

.chart {
  height: 400px;
}

.metric-card {
  text-align: center;
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.metric-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 2px 12px 0 rgba(0,0,0,.1);
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  margin: 10px 0;
  color: #409EFF;
}

.metric-trend {
  font-size: 14px;
  margin-top: 8px;
}

.metric-trend.positive {
  color: #67C23A;
}

.metric-trend.negative {
  color: #F56C6C;
}

.chart-row {
  margin-top: 20px;
}

.el-timeline {
  margin-top: 20px;
}

.el-timeline-item h4 {
  margin: 0;
  font-size: 16px;
  color: #303133;
}

.el-timeline-item p {
  margin: 8px 0;
  color: #606266;
}

.el-card {
  margin-bottom: 20px;
  transition: all 0.3s ease;
}

.el-card:hover {
  box-shadow: 0 4px 16px 0 rgba(0,0,0,.1);
}
</style>