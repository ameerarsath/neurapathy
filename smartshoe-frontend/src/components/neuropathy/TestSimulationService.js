/**
 * Neuropathy Test Simulation Service
 * Simulates actual hardware stimuli for demonstration purposes
 */

export class TestSimulationService {
  constructor() {
    this.isRunning = false
    this.currentStimulus = null
    this.onStimulusGenerated = null
    this.onTestComplete = null
    this.stimulusQueue = []
    this.currentIndex = 0
  }

  // Generate a randomized test sequence
  generateTestSequence(totalStimuli = 20) {
    const stimulusTypes = [
      { type: 'VIBRATION', name: 'Vibration', intensity: 0.3 + Math.random() * 0.7 },
      { type: 'TEMPERATURE_HOT', name: 'Hot Temperature', intensity: 0.4 + Math.random() * 0.6 },
      { type: 'TEMPERATURE_COLD', name: 'Cold Temperature', intensity: 0.3 + Math.random() * 0.7 },
      { type: 'PINPRICK', name: 'Sharp Sensation', intensity: 0.2 + Math.random() * 0.8 }
    ]

    const footLocations = [
      { region: 'heel', name: 'Heel', x: 40, y: 85 },
      { region: 'arch', name: 'Arch', x: 45, y: 60 },
      { region: 'ball', name: 'Ball of Foot', x: 50, y: 35 },
      { region: 'big-toe', name: 'Big Toe', x: 50, y: 15 },
      { region: 'toes', name: 'Toes', x: 35, y: 20 }
    ]

    this.stimulusQueue = []
    
    for (let i = 0; i < totalStimuli; i++) {
      // 15% chance of no-stimulus control trial
      const isControlTrial = Math.random() < 0.15
      
      if (isControlTrial) {
        this.stimulusQueue.push({
          id: i + 1,
          sequence: i + 1,
          type: 'NONE',
          name: 'No Stimulus',
          intensity: 0,
          location: null,
          isControlTrial: true,
          duration: 2000 + Math.random() * 3000 // 2-5 seconds
        })
      } else {
        const stimulus = stimulusTypes[Math.floor(Math.random() * stimulusTypes.length)]
        const location = footLocations[Math.floor(Math.random() * footLocations.length)]
        
        this.stimulusQueue.push({
          id: i + 1,
          sequence: i + 1,
          type: stimulus.type,
          name: stimulus.name,
          intensity: stimulus.intensity,
          location: location,
          isControlTrial: false,
          duration: 1000 + Math.random() * 2000 // 1-3 seconds
        })
      }
    }
    
    // Shuffle the array to randomize order
    this.stimulusQueue = this.shuffleArray(this.stimulusQueue)
    this.currentIndex = 0
  }

  shuffleArray(array) {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  }

  // Start the test simulation
  startTest(onStimulusGenerated, onTestComplete) {
    this.onStimulusGenerated = onStimulusGenerated
    this.onTestComplete = onTestComplete
    this.isRunning = true
    this.currentIndex = 0
    
    console.log('🧪 Starting neuropathy test simulation...')
    this.presentNextStimulus()
  }

  // Present the next stimulus to the patient
  presentNextStimulus() {
    if (!this.isRunning || this.currentIndex >= this.stimulusQueue.length) {
      this.completeTest()
      return
    }

    const stimulus = this.stimulusQueue[this.currentIndex]
    this.currentStimulus = stimulus
    
    console.log(`🎯 Presenting stimulus ${stimulus.sequence}: ${stimulus.name}`)
    
    // Simulate the stimulus presentation
    this.simulateStimulus(stimulus)
    
    // Notify the UI about the new stimulus
    if (this.onStimulusGenerated) {
      this.onStimulusGenerated(stimulus)
    }
  }

  // Simulate the actual stimulus (visual/audio feedback for demo)
  simulateStimulus(stimulus) {
    if (stimulus.isControlTrial) {
      console.log('🔇 Control trial - no stimulus presented')
      return
    }

    // Simulate different types of stimuli with visual/audio cues
    switch (stimulus.type) {
      case 'VIBRATION':
        this.simulateVibration(stimulus)
        break
      case 'TEMPERATURE_HOT':
        this.simulateHotTemperature(stimulus)
        break
      case 'TEMPERATURE_COLD':
        this.simulateColdTemperature(stimulus)
        break
      case 'PINPRICK':
        this.simulatePinprick(stimulus)
        break
      case 'PRESSURE':
        this.simulatePressure(stimulus)
        break
    }
  }

  simulateVibration(stimulus) {
    console.log(`📳 VIBRATION at ${stimulus.location.name} - Intensity: ${(stimulus.intensity * 10).toFixed(1)}/10`)
    
    // Visual feedback - could trigger UI animations
    if (typeof window !== 'undefined' && navigator.vibrate) {
      // Use device vibration if available
      const duration = Math.floor(stimulus.intensity * 200)
      navigator.vibrate([duration, 100, duration])
    }
  }

  simulateHotTemperature(stimulus) {
    console.log(`🔥 HOT TEMPERATURE at ${stimulus.location.name} - Intensity: ${(stimulus.intensity * 10).toFixed(1)}/10`)
    
    // Could trigger red color flash on UI
    this.triggerColorFlash('#ff4444', stimulus.duration)
  }

  simulateColdTemperature(stimulus) {
    console.log(`❄️ COLD TEMPERATURE at ${stimulus.location.name} - Intensity: ${(stimulus.intensity * 10).toFixed(1)}/10`)
    
    // Could trigger blue color flash on UI
    this.triggerColorFlash('#4444ff', stimulus.duration)
  }

  simulatePinprick(stimulus) {
    console.log(`📌 PINPRICK at ${stimulus.location.name} - Intensity: ${(stimulus.intensity * 10).toFixed(1)}/10`)
    
    // Could trigger sharp visual effect
    this.triggerColorFlash('#ffaa00', 200)
  }

  simulatePressure(stimulus) {
    console.log(`⚪ PRESSURE at ${stimulus.location.name} - Intensity: ${(stimulus.intensity * 10).toFixed(1)}/10`)
    
    // Could trigger pressure animation
    this.triggerColorFlash('#888888', stimulus.duration)
  }

  triggerColorFlash(color, duration) {
    // This could be used to flash the background or show visual indicators
    if (typeof document !== 'undefined') {
      const originalColor = document.body.style.backgroundColor
      document.body.style.backgroundColor = color
      document.body.style.opacity = '0.3'
      
      setTimeout(() => {
        document.body.style.backgroundColor = originalColor
        document.body.style.opacity = '1'
      }, Math.min(duration, 500))
    }
  }

  // Patient submits response to current stimulus
  submitResponse(response) {
    if (!this.currentStimulus || !this.isRunning) {
      console.warn('No active stimulus to respond to')
      return false
    }

    console.log(`✅ Response recorded for stimulus ${this.currentStimulus.sequence}:`, {
      feltSensation: response.feltSensation,
      perceivedIntensity: response.perceivedIntensity,
      perceivedType: response.perceivedType,
      confidence: response.responseConfidence
    })

    // Move to next stimulus
    this.currentIndex++
    
    // Wait a moment before presenting next stimulus
    setTimeout(() => {
      this.presentNextStimulus()
    }, 1000 + Math.random() * 2000) // 1-3 second delay

    return true
  }

  // Complete the test
  completeTest() {
    this.isRunning = false
    this.currentStimulus = null
    
    console.log('🎉 Neuropathy test simulation completed!')
    
    if (this.onTestComplete) {
      this.onTestComplete()
    }
  }

  // Stop the test early
  stopTest() {
    this.isRunning = false
    this.currentStimulus = null
    console.log('⏹️ Test simulation stopped')
  }

  // Get current progress
  getProgress() {
    return {
      current: this.currentIndex,
      total: this.stimulusQueue.length,
      percentage: (this.currentIndex / this.stimulusQueue.length) * 100
    }
  }

  // Get current stimulus info (without revealing actual values to patient)
  getCurrentStimulusForPatient() {
    if (!this.currentStimulus) return null
    
    return {
      id: this.currentStimulus.id,
      sequence: this.currentStimulus.sequence,
      isControlTrial: this.currentStimulus.isControlTrial,
      // Don't reveal actual stimulus details to patient
      message: this.currentStimulus.isControlTrial 
        ? "Focus and report any sensations you feel" 
        : "A stimulus may be presented - report what you feel"
    }
  }

  // Get current stimulus info for physician (includes actual stimulus data)
  getCurrentStimulusForPhysician() {
    return this.currentStimulus
  }
}

// Create singleton instance
export const testSimulation = new TestSimulationService()
export default testSimulation