# 🧠 ML Model Testing Guide

This guide will help you test the integrated ML models for neuropathy detection in the frontend.

## 🚀 Quick Start

### 1. Start the ML API

**Option A: Using Startup Scripts (Recommended)**
```bash
# For Linux/Mac
./start-ml-api.sh

# For Windows
start-ml-api.bat
```

**Option B: If you get dependency errors, use the minimal API**
```bash
cd ml-models
python src/deployment/minimal_api.py
```

**Option C: Fix dependencies first (if needed)**
```bash
# For Windows
setup-ml-env.bat

# For Linux/Mac
./setup-ml-env.sh
```

The ML API will start on `http://localhost:8000`

### 2. Start the Spring Boot Backend

```bash
cd backend
mvn spring-boot:run
```

The backend will start on `http://localhost:8080`

### 3. Start the Frontend

```bash
cd smartshoe-frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

## 🧪 Testing the ML Models

### Access the ML Testing Lab

1. **Login** as a provider (doctor/admin):
   - Username: `doctor` / Password: `doctor123`
   - Username: `admin` / Password: `admin123`

2. **Navigate** to "ML Testing Lab" in the sidebar

3. **Check ML Service Status** - Should show "healthy" if everything is connected

### Test Workflow

#### Step 1: Select Test Data
1. **Choose a Patient**: Select from the dropdown (e.g., John Doe)
2. **Select Reading** (optional): Choose a specific medical reading or use the latest

#### Step 2: Run Tests
1. **Individual Tests**: Click on any of the 4 model buttons:
   - 🧠 **Neuropathy Test**: Predict disease progression
   - 💚 **Glucose Test**: Assess diabetes complications
   - ⚡ **Anomaly Test**: Detect sensor irregularities
   - 📈 **Risk Test**: Calculate overall risk stratification

2. **Batch Testing**: Click "Run All Tests" to test all models simultaneously

#### Step 3: View Results
- **Real-time Results**: See immediate test outcomes with risk levels
- **Confidence Scores**: View model confidence percentages
- **Additional Data**: Explore model-specific insights
- **Historical Data**: Review previous predictions

## 🔍 Understanding Results

### Risk Levels
- 🔴 **HIGH** (>70%): Immediate attention required
- 🟡 **MEDIUM** (40-70%): Monitor closely
- 🟢 **LOW** (<40%): Normal range

### Confidence Scores
- 🟢 **HIGH** (>80%): Very reliable
- 🟡 **MEDIUM** (60-80%): Moderately reliable
- 🔴 **LOW** (<60%): Less reliable

### Model Types
1. **Neuropathy Progression**: Predicts advancement of nerve damage
2. **Glucose Complications**: Assesses diabetes-related risks
3. **Anomaly Detection**: Identifies unusual sensor patterns
4. **Risk Stratification**: Overall patient risk assessment

## 🛠 Troubleshooting

### Common Issues

#### ML Service Unhealthy
```
✅ Solution: 
1. Check if ML API is running on port 8000
2. Restart ML API using startup scripts
3. Check Python dependencies are installed
```

#### No Patients Available
```
✅ Solution:
1. Ensure Spring Boot backend is running
2. Check database initialization (sample data should be created)
3. Login with provider credentials (doctor/admin)
```

#### Test Failures
```
✅ Solution:
1. Check browser console for detailed errors
2. Verify backend logs for ML integration issues
3. Ensure patient has medical readings data
```

#### Authentication Errors
```
✅ Solution:
1. Login with provider role (doctor/admin)
2. Patient users cannot access ML testing
3. Check token in localStorage is valid
```

### Debug Checklist

- [ ] ML API running on port 8000 ✓
- [ ] Spring Boot backend running on port 8080 ✓
- [ ] Frontend running on port 3000 ✓
- [ ] Logged in as provider (doctor/admin) ✓
- [ ] Sample patients available in database ✓
- [ ] Medical readings exist for selected patient ✓

## 📊 Test Scenarios

### Scenario 1: Basic Model Testing
1. Select "John Doe" (Patient ID: 1)
2. Run "Neuropathy Test"
3. Verify result shows prediction percentage and confidence
4. Check risk level classification

### Scenario 2: Batch Analysis
1. Select any patient with medical readings
2. Click "Run All Tests"
3. Verify all 4 models complete successfully
4. Compare results across different model types

### Scenario 3: Historical Analysis
1. Run multiple tests on the same patient
2. Navigate to "Historical Predictions" section
3. Verify predictions are saved and retrievable
4. Check timestamps and model versions

### Scenario 4: Real-time Integration
1. Create a new medical reading via "Neuropathy Testing"
2. Check if ML analysis is triggered automatically
3. Verify predictions appear in ML Testing Lab
4. Confirm alerts are generated for high-risk results

## 🔧 Advanced Testing

### Custom Test Data
To test with specific sensor values:
1. Create custom medical readings via the API
2. Use different reading types (PRESSURE, VIBRATION, TEMPERATURE)
3. Test with various severity levels (NORMAL, MILD, MODERATE, SEVERE)

### API Testing
Direct ML API testing:
```bash
# Test ML API health
curl http://localhost:8000/health

# Test neuropathy prediction
curl -X POST "http://localhost:8000/predict/neuropathy-progression" \
  -H "Authorization: Bearer ml_api_dev_token" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "1",
    "model_type": "neuropathy_progression",
    "features": {
      "age": 65,
      "gender_encoded": 1,
      "diabetes_type_encoded": 2,
      "years_diabetes": 15
    }
  }'
```

### Performance Testing
- Test with multiple patients simultaneously
- Monitor response times for each model
- Check memory usage during batch processing
- Verify caching behavior

## 📈 Monitoring & Analytics

### Model Performance
- Access model metrics via ML Testing Lab
- Monitor accuracy, precision, and recall
- Track prediction latency
- Review confidence score distributions

### Usage Analytics
- Track number of predictions per model
- Monitor error rates and failure patterns
- Analyze user interaction patterns
- Review system performance metrics

## 🎯 Next Steps

After successful testing:
1. **Deploy to Production**: Configure production ML API endpoints
2. **Set Up Monitoring**: Implement comprehensive logging and alerting
3. **Train Custom Models**: Use your specific patient data
4. **Optimize Performance**: Fine-tune model parameters
5. **Expand Features**: Add more ML models and capabilities

## 📞 Support

If you encounter issues:
1. Check browser console for frontend errors
2. Review Spring Boot logs for backend issues
3. Monitor ML API logs for model problems
4. Use the health check endpoints to verify connectivity

---

Happy Testing! 🚀 Your ML-powered neuropathy detection system is ready for validation.