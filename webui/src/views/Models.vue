<script>
import { mapState, mapActions } from 'vuex'

export default {
  name: 'Models',
  data() {
    return {
      addModelDialogVisible: false,
      configDialogVisible: false,
      newModel: {
        name: '',
        type: '',
        config: null
      }
    }
  },
  computed: {
    ...mapState('models', ['models', 'selectedModel', 'loading', 'error'])
  },
  methods: {
    ...mapActions('models', ['fetchModels', 'addModel', 'updateModel', 'removeModel', 'toggleModelStatus']),
    showAddModelDialog() {
      this.addModelDialogVisible = true
    },
    showConfigDialog(model) {
      this.selectedModel = model
      this.configDialogVisible = true
    },
    async handleAddModel() {
      try {
        await this.addModel(this.newModel)
        this.addModelDialogVisible = false
        this.newModel = { name: '', type: '', config: null }
        this.$message.success('Model added successfully')
      } catch (error) {
        this.$message.error('Failed to add model: ' + error.message)
      }
    },
    async handleRemoveModel(model) {
      try {
        await this.$confirm('Are you sure you want to delete this model?')
        await this.removeModel(model.id)
        this.$message.success('Model removed successfully')
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('Failed to remove model: ' + error.message)
        }
      }
    },
    async handleToggleStatus(model) {
      try {
        await this.toggleModelStatus({
          id: model.id,
          active: model.status !== 'active'
        })
        this.$message.success(`Model ${model.status === 'active' ? 'stopped' : 'started'} successfully`)
      } catch (error) {
        this.$message.error('Failed to toggle model status: ' + error.message)
      }
    },
    format(percentage) {
      return percentage + '%'
    }
  },
  created() {
    this.fetchModels()
  }
}
</script>

<template>
  <div class="models">
    <el-row :gutter="20">
      <el-col :span="16">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>Loaded Models</span>
              <el-button type="primary" @click="showAddModelDialog">Add Model</el-button>
            </div>
          </template>
          <el-table :data="models" style="width: 100%">
            <el-table-column prop="name" label="Model Name" />
            <el-table-column prop="type" label="Type" />
            <el-table-column prop="status" label="Status">
              <template #default="{ row }">
                <el-tag :type="row.status === 'active' ? 'success' : 'info'">
                  {{ row.status === 'active' ? 'Running' : 'Stopped' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="200">
              <template #default="{ row }">
                <el-button
                  :type="row.status === 'active' ? 'warning' : 'success'"
                  size="small"
                  @click="toggleModelStatus(row)">
                  {{ row.status === 'active' ? 'Stop' : 'Start' }}
                </el-button>
                <el-button
                  type="primary"
                  size="small"
                  @click="showConfigDialog(row)">
                  Configure
                </el-button>
                <el-button
                  type="danger"
                  size="small"
                  @click="removeModel(row)">
                  Delete
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="model-info">
          <template #header>
            <div class="card-header">
              <span>Model Details</span>
            </div>
          </template>
          <div v-if="selectedModel">
            <el-descriptions :column="1">
              <el-descriptions-item label="Model Name">{{ selectedModel.name }}</el-descriptions-item>
              <el-descriptions-item label="Type">{{ selectedModel.type }}</el-descriptions-item>
              <el-descriptions-item label="Status">
                <el-tag :type="selectedModel.status === 'active' ? 'success' : 'info'">
                  {{ selectedModel.status === 'active' ? 'Running' : 'Stopped' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="Created At">{{ selectedModel.createdAt }}</el-descriptions-item>
              <el-descriptions-item label="Last Updated">{{ selectedModel.updatedAt }}</el-descriptions-item>
            </el-descriptions>
            <div class="model-metrics">
              <h4>Performance Metrics</h4>
              <el-progress
                :percentage="selectedModel.metrics.accuracy * 100"
                :format="format"
                type="dashboard"
                :stroke-width="8"
              />
            </div>
          </div>
          <el-empty v-else description="Please select a model to view details" />
        </el-card>
      </el-col>
    </el-row>

    <!-- Add Model Dialog -->
    <el-dialog
      v-model="addModelDialogVisible"
      title="Add New Model"
      width="500px">
      <el-form :model="newModel" label-width="100px">
        <el-form-item label="Model Name">
          <el-input v-model="newModel.name" />
        </el-form-item>
        <el-form-item label="Model Type">
          <el-select v-model="newModel.type" placeholder="Select model type">
            <el-option label="PolicyNetwork" value="policy" />
            <el-option label="ValueNetwork" value="value" />
            <el-option label="CustomModel" value="custom" />
          </el-select>
        </el-form-item>
        <el-form-item label="Config File">
          <el-upload
            class="upload-demo"
            action="/api/upload"
            :auto-upload="false">
            <template #trigger>
              <el-button type="primary">Select File</el-button>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="addModelDialogVisible = false">Cancel</el-button>
          <el-button type="primary" @click="addModel">Confirm</el-button>
        </span>
      </template>
    </el-dialog>

    <!-- Configure Model Dialog -->
    <el-dialog
      v-model="configDialogVisible"
      title="Model Configuration"
      width="500px">
      <el-form :model="modelConfig" label-width="100px">
        <el-form-item label="Learning Rate">
          <el-input-number
            v-model="modelConfig.learningRate"
            :precision="4"
            :step="0.0001"
            :min="0"
            :max="1"
          />
        </el-form-item>
        <el-form-item label="Batch Size">
          <el-input-number
            v-model="modelConfig.batchSize"
            :min="1"
            :max="512"
          />
        </el-form-item>
        <el-form-item label="Optimizer">
          <el-select v-model="modelConfig.optimizer">
            <el-option label="Adam" value="adam" />
            <el-option label="SGD" value="sgd" />
            <el-option label="RMSprop" value="rmsprop" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="configDialogVisible = false">Cancel</el-button>
          <el-button type="primary" @click="saveConfig">Save</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useStore } from 'vuex'
import { ElMessage } from 'element-plus'

const store = useStore()

// Model list data
const models = ref([
  {
    name: 'PolicyNetwork',
    type: 'policy',
    status: 'active',
    createdAt: '2023-08-15 10:30',
    updatedAt: '2023-08-15 15:45',
    metrics: {
      accuracy: 0.85
    }
  },
  {
    name: 'ValueEstimator',
    type: 'value',
    status: 'inactive',
    createdAt: '2023-08-14 09:20',
    updatedAt: '2023-08-15 11:30',
    metrics: {
      accuracy: 0.78
    }
  }
])

// Selected model
const selectedModel = ref(null)

// Add model related
const addModelDialogVisible = ref(false)
const newModel = ref({
  name: '',
  type: ''
})

// Configuration dialog related
const configDialogVisible = ref(false)
const modelConfig = ref({
  learningRate: 0.001,
  batchSize: 32,
  optimizer: 'adam'
})

// Format progress display
const format = (percentage) => percentage + '%'

// Show add model dialog
const showAddModelDialog = () => {
  addModelDialogVisible.value = true
  newModel.value = {
    name: '',
    type: ''
  }
}

// Add new model
const addModel = () => {
  if (!newModel.value.name || !newModel.value.type) {
    ElMessage.warning('Please fill in all model information')
    return
  }
  
  const model = {
    ...newModel.value,
    status: 'inactive',
    createdAt: new Date().toLocaleString(),
    updatedAt: new Date().toLocaleString(),
    metrics: {
      accuracy: 0
    }
  }
  
  models.value.push(model)
  addModelDialogVisible.value = false
  ElMessage.success('Model added successfully')
}

// Show configuration dialog
const showConfigDialog = (model) => {
  selectedModel.value = model
  configDialogVisible.value = true
}

// Save model configuration
const saveConfig = () => {
  // Should call API to save configuration
  configDialogVisible.value = false
  ElMessage.success('Configuration saved successfully')
}

// Toggle model status
const toggleModelStatus = (model) => {
  model.status = model.status === 'active' ? 'inactive' : 'active'
  model.updatedAt = new Date().toLocaleString()
  ElMessage.success(`Model ${model.status === 'active' ? 'started' : 'stopped'} successfully`)
}

// Remove model
const removeModel = (model) => {
  const index = models.value.indexOf(model)
  if (index > -1) {
    models.value.splice(index, 1)
    if (selectedModel.value === model) {
      selectedModel.value = null
    }
    ElMessage.success('Model deleted successfully')
  }
}
</script>

<style scoped>
.models {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.model-info {
  height: 100%;
}

.model-metrics {
  margin-top: 20px;
  text-align: center;
}

.model-metrics h4 {
  margin-bottom: 15px;
}

.el-progress {
  margin: 0 auto;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>