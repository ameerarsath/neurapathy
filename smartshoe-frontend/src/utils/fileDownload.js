/**
 * Utility functions for handling file downloads
 */

/**
 * Download a blob as a file
 * @param {Blob} blob - The blob to download
 * @param {string} filename - The filename to save as
 */
export const downloadBlob = (blob, filename) => {
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

/**
 * Download text content as a file
 * @param {string} content - The text content to download
 * @param {string} filename - The filename to save as
 * @param {string} type - The MIME type (default: 'text/plain')
 */
export const downloadText = (content, filename, type = 'text/plain') => {
  const blob = new Blob([content], { type })
  downloadBlob(blob, filename)
}

/**
 * Generate a filename with timestamp
 * @param {string} prefix - The filename prefix
 * @param {string} extension - The file extension
 * @returns {string} The generated filename
 */
export const generateFilename = (prefix, extension) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  return `${prefix}_${timestamp}.${extension}`
}

/**
 * Handle API response for file download
 * @param {Response} response - The API response
 * @param {string} defaultFilename - Default filename if not provided in response
 */
export const handleFileDownload = (response, defaultFilename) => {
  // Get filename from Content-Disposition header if available
  const contentDisposition = response.headers['content-disposition']
  let filename = defaultFilename
  
  if (contentDisposition) {
    const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/)
    if (filenameMatch) {
      filename = filenameMatch[1]
    }
  }
  
  downloadBlob(response.data, filename)
}