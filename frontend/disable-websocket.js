// Temporary script to disable WebSocket connections
// Run this in browser console to stop WebSocket connection attempts

console.log('🚫 Disabling WebSocket connections...');
localStorage.setItem('disableWebSocket', 'true');
window.location.reload();
console.log('✅ WebSocket connections disabled. Page reloaded.');