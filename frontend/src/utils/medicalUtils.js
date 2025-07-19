/**
 * Medical utility functions for neuropathy assessment and risk calculation
 */

export const calculateRiskLevel = (testResults, patientData) => {
  if (!testResults || !patientData) return 'unknown'
  
  let riskScore = 0
  const factors = []
  
  // Age factor (diabetes duration)
  if (patientData.diabetesDuration) {
    if (patientData.diabetesDuration > 15) {
      riskScore += 30
      factors.push('Long diabetes duration (>15 years)')
    } else if (patientData.diabetesDuration > 10) {
      riskScore += 20
      factors.push('Moderate diabetes duration (10-15 years)')
    }
  }
  
  // HbA1c levels
  if (patientData.hba1c) {
    if (patientData.hba1c > 9) {
      riskScore += 25
      factors.push('Poor glycemic control (HbA1c >9%)')
    } else if (patientData.hba1c > 7) {
      riskScore += 15
      factors.push('Suboptimal glycemic control (HbA1c 7-9%)')
    }
  }
  
  // Test results analysis
  if (testResults.vibrationThreshold) {
    if (testResults.vibrationThreshold > 25) {
      riskScore += 20
      factors.push('Elevated vibration threshold')
    }
  }
  
  if (testResults.pressureThreshold) {
    if (testResults.pressureThreshold > 10) {
      riskScore += 15
      factors.push('Reduced pressure sensitivity')
    }
  }
  
  if (testResults.temperatureThreshold) {
    if (testResults.temperatureThreshold > 5) {
      riskScore += 10
      factors.push('Impaired temperature sensation')
    }
  }
  
  // BMI factor
  if (patientData.bmi && patientData.bmi > 30) {
    riskScore += 10
    factors.push('Obesity (BMI >30)')
  }
  
  // Determine risk level
  let riskLevel = 'low'
  if (riskScore >= 70) {
    riskLevel = 'high'
  } else if (riskScore >= 40) {
    riskLevel = 'medium'
  }
  
  return {
    level: riskLevel,
    score: riskScore,
    factors,
    recommendations: getRecommendations(riskLevel, factors)
  }
}

export const getRiskColor = (riskLevel) => {
  if (typeof riskLevel === 'number') {
    if (riskLevel >= 70) return 'red'
    if (riskLevel >= 40) return 'yellow'
    return 'green'
  }
  
  switch (riskLevel?.toLowerCase()) {
    case 'high': return 'red'
    case 'medium': return 'yellow'
    case 'low': return 'green'
    default: return 'gray'
  }
}

export const getNeuropathySeverity = (testResults) => {
  if (!testResults) return 'Unknown'
  
  let severity = 0
  
  // Vibration test
  if (testResults.vibrationThreshold > 30) severity += 3
  else if (testResults.vibrationThreshold > 20) severity += 2
  else if (testResults.vibrationThreshold > 10) severity += 1
  
  // Pressure test
  if (testResults.pressureThreshold > 15) severity += 3
  else if (testResults.pressureThreshold > 10) severity += 2
  else if (testResults.pressureThreshold > 6) severity += 1
  
  // Temperature test
  if (testResults.temperatureThreshold > 8) severity += 3
  else if (testResults.temperatureThreshold > 5) severity += 2
  else if (testResults.temperatureThreshold > 3) severity += 1
  
  if (severity === 0) return 'Normal'
  if (severity <= 3) return 'Mild'
  if (severity <= 6) return 'Moderate'
  return 'Severe'
}

export const getRecommendations = (riskLevel, factors) => {
  const recommendations = []
  
  switch (riskLevel) {
    case 'high':
      recommendations.push('Schedule immediate consultation with healthcare provider')
      recommendations.push('Increase testing frequency to weekly')
      recommendations.push('Review medication adherence and dosages')
      recommendations.push('Consider referral to endocrinologist')
      break
    case 'medium':
      recommendations.push('Schedule follow-up appointment within 2 weeks')
      recommendations.push('Increase testing frequency to bi-weekly')
      recommendations.push('Review lifestyle and dietary habits')
      recommendations.push('Monitor blood glucose levels more frequently')
      break
    case 'low':
      recommendations.push('Continue regular testing schedule')
      recommendations.push('Maintain current treatment plan')
      recommendations.push('Regular exercise and healthy diet')
      recommendations.push('Monitor for any changes in sensation')
      break
  }
  
  // Factor-specific recommendations
  if (factors.includes('Poor glycemic control')) {
    recommendations.push('Focus on improved glucose management')
    recommendations.push('Consider insulin therapy adjustment')
  }
  
  if (factors.includes('Obesity')) {
    recommendations.push('Weight management program recommended')
    recommendations.push('Nutritionist consultation advised')
  }
  
  return recommendations
}

export const interpretTestResult = (testType, value, normalRange) => {
  if (!value || !normalRange) return { status: 'unknown', interpretation: 'Unable to interpret' }
  
  const { min, max } = normalRange
  
  if (value < min) {
    return {
      status: 'below_normal',
      interpretation: `${testType} sensitivity is higher than normal`,
      severity: 'mild'
    }
  } else if (value > max) {
    let severity = 'mild'
    if (value > max * 2) severity = 'severe'
    else if (value > max * 1.5) severity = 'moderate'
    
    return {
      status: 'above_normal',
      interpretation: `${testType} sensitivity is reduced`,
      severity
    }
  } else {
    return {
      status: 'normal',
      interpretation: `${testType} sensitivity is within normal range`,
      severity: 'none'
    }
  }
}

export const calculateProgressionRate = (currentResults, previousResults) => {
  if (!currentResults || !previousResults) return null
  
  const timeDiff = new Date(currentResults.date) - new Date(previousResults.date)
  const daysDiff = timeDiff / (1000 * 60 * 60 * 24)
  
  if (daysDiff <= 0) return null
  
  const improvements = []
  const deteriorations = []
  
  // Compare vibration threshold
  if (currentResults.vibrationThreshold && previousResults.vibrationThreshold) {
    const change = currentResults.vibrationThreshold - previousResults.vibrationThreshold
    if (change > 2) deteriorations.push('vibration')
    else if (change < -2) improvements.push('vibration')
  }
  
  // Compare pressure threshold
  if (currentResults.pressureThreshold && previousResults.pressureThreshold) {
    const change = currentResults.pressureThreshold - previousResults.pressureThreshold
    if (change > 1) deteriorations.push('pressure')
    else if (change < -1) improvements.push('pressure')
  }
  
  // Compare temperature threshold
  if (currentResults.temperatureThreshold && previousResults.temperatureThreshold) {
    const change = currentResults.temperatureThreshold - previousResults.temperatureThreshold
    if (change > 1) deteriorations.push('temperature')
    else if (change < -1) improvements.push('temperature')
  }
  
  let trend = 'stable'
  if (deteriorations.length > improvements.length) {
    trend = 'worsening'
  } else if (improvements.length > deteriorations.length) {
    trend = 'improving'
  }
  
  return {
    trend,
    improvements,
    deteriorations,
    daysSinceLastTest: Math.round(daysDiff),
    changeRate: (deteriorations.length - improvements.length) / daysDiff
  }
}

export const getTestFrequencyRecommendation = (riskLevel, lastTestDate) => {
  const now = new Date()
  const lastTest = new Date(lastTestDate)
  const daysSinceLastTest = (now - lastTest) / (1000 * 60 * 60 * 24)
  
  let recommendedFrequency
  switch (riskLevel) {
    case 'high':
      recommendedFrequency = 7 // weekly
      break
    case 'medium':
      recommendedFrequency = 14 // bi-weekly
      break
    case 'low':
    default:
      recommendedFrequency = 30 // monthly
      break
  }
  
  const isOverdue = daysSinceLastTest > recommendedFrequency
  const nextTestDate = new Date(lastTest)
  nextTestDate.setDate(nextTestDate.getDate() + recommendedFrequency)
  
  return {
    frequency: recommendedFrequency,
    isOverdue,
    daysSinceLastTest: Math.round(daysSinceLastTest),
    nextTestDate,
    urgency: isOverdue ? (daysSinceLastTest > recommendedFrequency * 2 ? 'high' : 'medium') : 'low'
  }
}

export const formatNeuropathyScore = (score) => {
  if (!score && score !== 0) return 'N/A'
  return `${Math.round(score * 10) / 10}/100`
}

export const getMedicalAlertLevel = (value, type) => {
  const thresholds = {
    glucose: { low: 70, normal: [80, 180], high: 250 },
    hba1c: { normal: [4, 7], elevated: [7, 9], high: 9 },
    blood_pressure: { normal: [90, 140], elevated: [140, 160], high: 180 },
    heart_rate: { normal: [60, 100], elevated: [100, 120], high: 150 }
  }
  
  const threshold = thresholds[type]
  if (!threshold) return 'unknown'
  
  if (type === 'glucose') {
    if (value < threshold.low) return 'critical_low'
    if (value > threshold.high) return 'critical_high'
    if (value < threshold.normal[0] || value > threshold.normal[1]) return 'warning'
    return 'normal'
  }
  
  if (type === 'hba1c') {
    if (value > threshold.high) return 'critical'
    if (value > threshold.elevated[0]) return 'warning'
    return 'normal'
  }
  
  if (Array.isArray(threshold.normal)) {
    if (value > threshold.high) return 'critical'
    if (value > threshold.elevated[0]) return 'warning'
    if (value < threshold.normal[0] || value > threshold.normal[1]) return 'attention'
    return 'normal'
  }
  
  return 'normal'
}