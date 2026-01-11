// document.addEventListener("DOMContentLoaded", () => {
//   const startScanBtn = document.getElementById("start-scan-btn")
//   const uploadSection = document.getElementById("upload-section")
//   const resultsSection = document.getElementById("results-section")
//   const dropArea = document.getElementById("drop-area")
//   const fileInput = document.getElementById("fileInput")
//   const gallery = document.getElementById("gallery")
//   const selectedCount = document.getElementById("selected-count")
//   const clearBtn = document.getElementById("clear-btn")
//   const analyzeBtn = document.getElementById("analyze-btn")
//   const backBtn = document.getElementById("back-btn")
//   const resultsLoading = document.getElementById("results-loading")
//   const resultsGrid = document.getElementById("results-grid")
//   const totalImages = document.getElementById("total-images")
//   const highRisk = document.getElementById("high-risk")
//   const avgTime = document.getElementById("avg-time")
//   const searchResults = document.getElementById("search-results")
//   const filterResults = document.getElementById("filter-results")
//   const exportBtn = document.getElementById("export-btn")
//   const detailModal = document.getElementById("detail-modal")
//   const closeModal = document.querySelector(".close-modal")
//   const askAiBtn = document.getElementById("ask-ai-btn")

//   // Variables
//   let selectedFiles = []
//   let results = []
//   let currentResult = null

//   // Event Listeners
//   startScanBtn.addEventListener("click", () => {
//     uploadSection.scrollIntoView({ behavior: "smooth" })
//   })

//   // Drag and drop functionality
//   ;["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
//     dropArea.addEventListener(eventName, preventDefaults, false)
//   })

//   function preventDefaults(e) {
//     e.preventDefault()
//     e.stopPropagation()
//   }
//   ;["dragenter", "dragover"].forEach((eventName) => {
//     dropArea.addEventListener(eventName, highlight, false)
//   })
//   ;["dragleave", "drop"].forEach((eventName) => {
//     dropArea.addEventListener(eventName, unhighlight, false)
//   })

//   function highlight() {
//     dropArea.classList.add("highlight")
//   }

//   function unhighlight() {
//     dropArea.classList.remove("highlight")
//   }

//   dropArea.addEventListener("drop", handleDrop, false)

//   function handleDrop(e) {
//     const dt = e.dataTransfer
//     const files = dt.files
//     handleFiles(files)
//   }

//   fileInput.addEventListener("change", function () {
//     handleFiles(this.files)
//   })

//   function handleFiles(files) {
//     if (files.length > 0) {
//       const newFiles = Array.from(files).filter((file) => file.type.startsWith("image/"))

//       // Check if adding these files would exceed the limit
//       if (selectedFiles.length + newFiles.length > 200) {
//         alert("You can upload a maximum of 200 images.")
//         return
//       }

//       // Add new files to selected files
//       selectedFiles = [...selectedFiles, ...newFiles]
//       updateFileCount()
//       updateGallery()
//       updateButtons()
//     }
//   }

//   function updateFileCount() {
//     selectedCount.textContent = selectedFiles.length
//   }

//   function updateGallery() {
//     gallery.innerHTML = ""

//     selectedFiles.forEach((file, index) => {
//       const reader = new FileReader()

//       reader.onload = (e) => {
//         const galleryItem = document.createElement("div")
//         galleryItem.className = "gallery-item"

//         const img = document.createElement("img")
//         img.src = e.target.result
//         img.alt = file.name

//         const removeBtn = document.createElement("button")
//         removeBtn.className = "remove-btn"
//         removeBtn.innerHTML = '<i class="fas fa-times"></i>'
//         removeBtn.addEventListener("click", () => removeFile(index))

//         galleryItem.appendChild(img)
//         galleryItem.appendChild(removeBtn)
//         gallery.appendChild(galleryItem)
//       }

//       reader.readAsDataURL(file)
//     })
//   }

//   function removeFile(index) {
//     selectedFiles.splice(index, 1)
//     updateFileCount()
//     updateGallery()
//     updateButtons()
//   }

//   function updateButtons() {
//     if (selectedFiles.length > 0) {
//       clearBtn.disabled = false
//       analyzeBtn.disabled = false
//     } else {
//       clearBtn.disabled = true
//       analyzeBtn.disabled = true
//     }
//   }

//   clearBtn.addEventListener("click", () => {
//     selectedFiles = []
//     updateFileCount()
//     updateGallery()
//     updateButtons()
//   })

//   analyzeBtn.addEventListener("click", () => {
//     // Show results section and hide upload section
//     uploadSection.classList.add("hidden")
//     resultsSection.classList.remove("hidden")

//     // Show loading
//     resultsLoading.classList.remove("hidden")
//     resultsGrid.innerHTML = ""

//     // Create form data
//     const formData = new FormData()
//     selectedFiles.forEach((file) => {
//       formData.append("files[]", file)
//     })

//     // Send request to server
//     fetch("/predict", {
//       method: "POST",
//       body: formData,
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         // Hide loading
//         resultsLoading.classList.add("hidden")

//         // Store results
//         results = data.results

//         // Update summary
//         totalImages.textContent = results.length

//         // Count high risk results
//         const highRiskCount = results.filter(
//           (result) => result.risk_level === "High" || result.risk_level === "Very High",
//         ).length
//         highRisk.textContent = highRiskCount

//         // Calculate average processing time
//         const avgProcessingTime =
//           results.reduce((sum, result) => sum + (result.processing_time || 0), 0) / results.length
//         avgTime.textContent = avgProcessingTime.toFixed(2) + "s"

//         // Display results
//         displayResults(results)

//         // Automatically generate and show summary report in chatbot
//         generateSummaryReport(results)
//       })
//       .catch((error) => {
//         console.error("Error:", error)
//         resultsLoading.classList.add("hidden")
//         alert("An error occurred while processing the images. Please try again.")
//       })
//   })

//   function displayResults(resultsArray) {
//     resultsGrid.innerHTML = ""

//     resultsArray.forEach((result) => {
//       if (result.error) {
//         // Handle error case
//         const errorCard = document.createElement("div")
//         errorCard.className = "result-card"
//         errorCard.innerHTML = `
//                     <div class="result-info">
//                         <h3>Error Processing Image</h3>
//                         <p>${result.original_filename}</p>
//                         <p class="error-message">${result.error}</p>
//                     </div>
//                 `
//         resultsGrid.appendChild(errorCard)
//         return
//       }

//       // Create result card
//       const resultCard = document.createElement("div")
//       resultCard.className = "result-card"
//       resultCard.dataset.result = JSON.stringify(result)

//       // Determine risk class
//       let riskClass = "low"
//       if (result.risk_level === "Moderate") riskClass = "moderate"
//       if (result.risk_level === "High") riskClass = "high"
//       if (result.risk_level === "Very High") riskClass = "very-high"

//       resultCard.innerHTML = `
//                 <div class="result-image">
//                     <img src="/uploads/${result.filename}" alt="${result.original_filename}">
//                 </div>
//                 <div class="result-info">
//                     <h3>${result.predicted_class}</h3>
//                     <p>${result.original_filename}</p>
//                     <div class="result-meta">
//                         <span class="risk-badge ${riskClass}">${result.risk_level}</span>
//                         <span class="confidence">${Math.round(result.confidence * 100)}% confidence</span>
//                     </div>
//                 </div>
//             `

//       // Add click event to show details
//       resultCard.addEventListener("click", () => showDetails(result))

//       resultsGrid.appendChild(resultCard)
//     })

//     // Make the view summary button visible
//     const viewSummaryBtn = document.getElementById("view-summary-btn")
//     if (viewSummaryBtn) {
//       viewSummaryBtn.style.display = "flex"
//     }
//   }

//   function showDetails(result) {
//     // Store current result for chatbot context
//     currentResult = result

//     // Populate modal with result data
//     document.getElementById("detail-img").src = `/uploads/${result.filename}`
//     document.getElementById("detail-diagnosis").textContent = result.predicted_class
//     document.getElementById("confidence-value").textContent = `${Math.round(result.confidence * 100)}%`
//     document.getElementById("confidence-bar").style.width = `${result.confidence * 100}%`

//     const riskElement = document.getElementById("risk-level")
//     riskElement.textContent = result.risk_level
//     riskElement.className = "risk-badge"

//     // Add risk class
//     if (result.risk_level === "Low") riskElement.classList.add("low")
//     if (result.risk_level === "Moderate") riskElement.classList.add("moderate")
//     if (result.risk_level === "High") riskElement.classList.add("high")
//     if (result.risk_level === "Very High") riskElement.classList.add("very-high")

//     document.getElementById("description-text").textContent = result.description
//     document.getElementById("detail-chart").src = `data:image/png;base64,${result.visualization}`

//     // Show modal
//     detailModal.style.display = "block"

//     // Set context for chatbot
//     if (window.setChatbotContext) {
//       window.setChatbotContext(result)
//     }
//   }

//   // Close modal when clicking the X
//   closeModal.addEventListener("click", () => {
//     detailModal.style.display = "none"
//   })

//   // Close modal when clicking outside the content
//   window.addEventListener("click", (e) => {
//     if (e.target === detailModal) {
//       detailModal.style.display = "none"
//     }
//   })

//   // Ask AI button
//   if (askAiBtn) {
//     askAiBtn.addEventListener("click", () => {
//       // Open chatbot with context
//       openChatbot(currentResult)
//     })
//   }

//   backBtn.addEventListener("click", () => {
//     resultsSection.classList.add("hidden")
//     uploadSection.classList.remove("hidden")
//   })

//   // Search and filter functionality
//   searchResults.addEventListener("input", filterResultsDisplay)
//   filterResults.addEventListener("change", filterResultsDisplay)

//   function filterResultsDisplay() {
//     const searchTerm = searchResults.value.toLowerCase()
//     const filterValue = filterResults.value

//     let filteredResults = [...results]

//     // Apply search filter
//     if (searchTerm) {
//       filteredResults = filteredResults.filter(
//         (result) =>
//           result.original_filename.toLowerCase().includes(searchTerm) ||
//           result.predicted_class.toLowerCase().includes(searchTerm),
//       )
//     }

//     // Apply risk filter
//     if (filterValue === "high-risk") {
//       filteredResults = filteredResults.filter(
//         (result) => result.risk_level === "High" || result.risk_level === "Very High",
//       )
//     } else if (filterValue === "low-risk") {
//       filteredResults = filteredResults.filter(
//         (result) => result.risk_level === "Low" || result.risk_level === "Moderate",
//       )
//     }

//     displayResults(filteredResults)
//   }

//   // Export results functionality
//   exportBtn.addEventListener("click", () => {
//     // Create CSV content
//     let csvContent = "data:text/csv;charset=utf-8,"
//     csvContent += "Filename,Diagnosis,Confidence,Risk Level\n"

//     results.forEach((result) => {
//       if (!result.error) {
//         csvContent += `${result.original_filename},${result.predicted_class},${Math.round(result.confidence * 100)}%,${result.risk_level}\n`
//       }
//     })

//     // Create download link
//     const encodedUri = encodeURI(csvContent)
//     const link = document.createElement("a")
//     link.setAttribute("href", encodedUri)
//     link.setAttribute("download", "skin_analysis_results.csv")
//     document.body.appendChild(link)

//     // Trigger download
//     link.click()
//     document.body.removeChild(link)
//   })

//   // Function to open chatbot with context
//   function openChatbot(result) {
//     const chatbotModal = document.getElementById("chatbot-modal")
//     if (chatbotModal) {
//       chatbotModal.style.display = "block"

//       // If we have a result, add a welcome message with context
//       if (result) {
//         const chatbotMessages = document.getElementById("chatbot-messages")
//         chatbotMessages.innerHTML = `
//                     <div class="message bot-message">
//                         <div class="message-content">
//                             <p>Hello! I'm your SkinScan AI Assistant. I can help explain your diagnosis of ${result.predicted_class} and answer any questions you might have about it. What would you like to know?</p>
//                         </div>
//                     </div>
//                 `
//       }
//     }
//   }

//   // Function to generate and display summary report in chatbot
//   // Add the view summary button event listener
//   const viewSummaryBtn = document.getElementById("view-summary-btn")
//   if (viewSummaryBtn) {
//     viewSummaryBtn.addEventListener("click", () => {
//       generateSummaryReport(results)
//     })
//   }

//   // Update the generateSummaryReport function to create a more comprehensive report
//   function generateSummaryReport(results) {
//     if (results.length === 0) return

//     // Count by diagnosis
//     const diagnosisCounts = {}
//     results.forEach((result) => {
//       if (!result.error) {
//         const diagnosis = result.predicted_class
//         diagnosisCounts[diagnosis] = (diagnosisCounts[diagnosis] || 0) + 1
//       }
//     })

//     // Count by risk level
//     const riskCounts = {
//       Low: 0,
//       Moderate: 0,
//       High: 0,
//       "Very High": 0,
//     }

//     results.forEach((result) => {
//       if (!result.error) {
//         riskCounts[result.risk_level] = (riskCounts[result.risk_level] || 0) + 1
//       }
//     })

//     // Find highest confidence result
//     let highestConfidence = 0
//     let highestConfidenceResult = null

//     results.forEach((result) => {
//       if (!result.error && result.confidence > highestConfidence) {
//         highestConfidence = result.confidence
//         highestConfidenceResult = result
//       }
//     })

//     // Find highest risk results
//     const highRiskResults = results.filter(
//       (result) => !result.error && (result.risk_level === "High" || result.risk_level === "Very High"),
//     )

//     // Generate summary HTML
//     let summaryHTML = `
//       <div class="summary-report">
//         <h2>Analysis Summary Report</h2>
        
//         <p>We've analyzed ${results.length} image${results.length !== 1 ? "s" : ""} and here's a comprehensive summary of our findings:</p>
        
//         <div class="risk-summary">
//           <div class="risk-item">
//             <h4>Total Images</h4>
//             <p>${results.length}</p>
//           </div>
//           <div class="risk-item">
//             <h4>High Risk Findings</h4>
//             <p>${riskCounts["High"] + riskCounts["Very High"]}</p>
//           </div>
//           <div class="risk-item">
//             <h4>Moderate Risk</h4>
//             <p>${riskCounts["Moderate"]}</p>
//           </div>
//           <div class="risk-item">
//             <h4>Low Risk</h4>
//             <p>${riskCounts["Low"]}</p>
//           </div>
//         </div>

//         <h3>Risk Level Breakdown</h3>
//         <ul>
//     `

//     if (riskCounts["Very High"] > 0) {
//       summaryHTML += `<li><strong>Very High Risk:</strong> ${riskCounts["Very High"]} image${riskCounts["Very High"] !== 1 ? "s" : ""}</li>`
//     }
//     if (riskCounts["High"] > 0) {
//       summaryHTML += `<li><strong>High Risk:</strong> ${riskCounts["High"]} image${riskCounts["High"] !== 1 ? "s" : ""}</li>`
//     }
//     if (riskCounts["Moderate"] > 0) {
//       summaryHTML += `<li><strong>Moderate Risk:</strong> ${riskCounts["Moderate"]} image${riskCounts["Moderate"] !== 1 ? "s" : ""}</li>`
//     }
//     if (riskCounts["Low"] > 0) {
//       summaryHTML += `<li><strong>Low Risk:</strong> ${riskCounts["Low"]} image${riskCounts["Low"] !== 1 ? "s" : ""}</li>`
//     }

//     summaryHTML += `
//         </ul>
        
//         <h3>Diagnosis Breakdown</h3>
//         <ul>
//     `

//     Object.keys(diagnosisCounts).forEach((diagnosis) => {
//       summaryHTML += `<li><strong>${diagnosis}:</strong> ${diagnosisCounts[diagnosis]} image${diagnosisCounts[diagnosis] !== 1 ? "s" : ""}</li>`
//     })

//     summaryHTML += `
//         </ul>
//     `

//     // Add high risk findings section if applicable
//     if (highRiskResults.length > 0) {
//       summaryHTML += `
//         <h3>High Risk Findings</h3>
//         <p>The following high-risk conditions were detected and require attention:</p>
//         <ul>
//       `

//       highRiskResults.forEach((result) => {
//         summaryHTML += `
//           <li>
//             <strong>${result.predicted_class}</strong> (${Math.round(result.confidence * 100)}% confidence) - 
//             ${result.description}
//           </li>
//         `
//       })

//       summaryHTML += `
//         </ul>
//       `
//     }

//     // Most confident diagnosis
//     if (highestConfidenceResult) {
//       summaryHTML += `
//         <h3>Most Confident Diagnosis</h3>
//         <p><strong>${highestConfidenceResult.predicted_class}</strong> (${Math.round(highestConfidenceResult.confidence * 100)}% confidence)</p>
//         <p><strong>Risk Level:</strong> ${highestConfidenceResult.risk_level}</p>
//         <p><strong>Description:</strong> ${highestConfidenceResult.description}</p>
//       `
//     }

//     // Add recommendations section
//     summaryHTML += `
//       <div class="recommendations">
//         <h3>Recommendations</h3>
//         <ul>
//     `

//     if (riskCounts["Very High"] > 0) {
//       summaryHTML += `
//         <li><strong>URGENT:</strong> Please consult with a dermatologist as soon as possible about the very high-risk findings. These conditions can be serious and require prompt medical attention.</li>
//       `
//     } else if (riskCounts["High"] > 0) {
//       summaryHTML += `
//         <li><strong>IMPORTANT:</strong> Schedule an appointment with a dermatologist within the next few weeks to evaluate the high-risk findings. Early treatment is key for better outcomes.</li>
//       `
//     } else if (riskCounts["Moderate"] > 0) {
//       summaryHTML += `
//         <li><strong>RECOMMENDED:</strong> Schedule a routine appointment with a dermatologist to evaluate the moderate-risk findings. While not urgent, these should be professionally assessed.</li>
//       `
//     } else {
//       summaryHTML += `
//         <li><strong>ROUTINE:</strong> Consider discussing these findings during your next regular check-up. Low-risk findings generally don't require immediate attention but should be monitored.</li>
//       `
//     }

//     summaryHTML += `
//         <li><strong>MONITORING:</strong> Continue to monitor your skin for any changes using the ABCDE method:
//           <ul>
//             <li><strong>A</strong>symmetry: One half doesn't match the other</li>
//             <li><strong>B</strong>order irregularity: Edges are ragged or blurred</li>
//             <li><strong>C</strong>olor variations: Multiple colors within the same lesion</li>
//             <li><strong>D</strong>iameter: Larger than 6mm (pencil eraser size)</li>
//             <li><strong>E</strong>volving: Changes in size, shape, color, or symptoms</li>
//           </ul>
//         </li>
//         <li><strong>PREVENTION:</strong> Practice sun safety with SPF 30+ sunscreen, protective clothing, and limiting exposure during peak hours (10am-4pm).</li>
//         <li><strong>FOLLOW-UP:</strong> Consider annual skin checks with a dermatologist, especially if you have risk factors like fair skin, history of sunburns, or family history of skin cancer.</li>
//       </ul>
//       </div>
      
//       <div class="disclaimer">
//         <p>IMPORTANT: This is an AI-generated analysis and should not replace professional medical advice. Always consult with a qualified healthcare provider for proper evaluation and treatment of skin conditions.</p>
//       </div>
//     </div>
//     `

//     // Open modal and display summary
//     const chatbotModal = document.getElementById("chatbot-modal")
//     if (chatbotModal) {
//       chatbotModal.style.display = "block"

//       const chatbotMessages = document.getElementById("chatbot-messages")
//       chatbotMessages.innerHTML = summaryHTML
//     }
//   }

//   // Expose the function to the global scope
//   window.openChatbot = openChatbot
//   window.generateSummaryReport = generateSummaryReport
// })


document.addEventListener("DOMContentLoaded", () => {
  const startScanBtn = document.getElementById("start-scan-btn")
  const uploadSection = document.getElementById("upload-section")
  const resultsSection = document.getElementById("results-section")
  const dropArea = document.getElementById("drop-area")
  const fileInput = document.getElementById("fileInput")
  const gallery = document.getElementById("gallery")
  const selectedCount = document.getElementById("selected-count")
  const clearBtn = document.getElementById("clear-btn")
  const analyzeBtn = document.getElementById("analyze-btn")
  const backBtn = document.getElementById("back-btn")
  const resultsLoading = document.getElementById("results-loading")
  const resultsGrid = document.getElementById("results-grid")
  const totalImages = document.getElementById("total-images")
  const highRisk = document.getElementById("high-risk")
  const avgTime = document.getElementById("avg-time")
  const searchResults = document.getElementById("search-results")
  const filterResults = document.getElementById("filter-results")
  const exportBtn = document.getElementById("export-btn")
  const detailModal = document.getElementById("detail-modal")
  const closeModal = document.querySelector(".close-modal")
  const askAiBtn = document.getElementById("ask-ai-btn")

  // Variables
  let selectedFiles = []
  let results = []
  let currentResult = null

  // Event Listeners
  startScanBtn.addEventListener("click", () => {
    uploadSection.scrollIntoView({ behavior: "smooth" })
  })

  // Drag and drop functionality
  ;["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false)
  })

  function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
  }
  ;["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false)
  })
  ;["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false)
  })

  function highlight() {
    dropArea.classList.add("highlight")
  }

  function unhighlight() {
    dropArea.classList.remove("highlight")
  }

  dropArea.addEventListener("drop", handleDrop, false)

  function handleDrop(e) {
    const dt = e.dataTransfer
    const files = dt.files
    handleFiles(files)
  }

  fileInput.addEventListener("change", function () {
    handleFiles(this.files)
  })

  function handleFiles(files) {
    if (files.length > 0) {
      const newFiles = Array.from(files).filter((file) => file.type.startsWith("image/"))

      // Check if adding these files would exceed the limit
      if (selectedFiles.length + newFiles.length > 200) {
        alert("You can upload a maximum of 200 images.")
        return
      }

      // Add new files to selected files
      selectedFiles = [...selectedFiles, ...newFiles]
      updateFileCount()
      updateGallery()
      updateButtons()
    }
  }

  function updateFileCount() {
    selectedCount.textContent = selectedFiles.length
  }

  function updateGallery() {
    gallery.innerHTML = ""

    selectedFiles.forEach((file, index) => {
      const reader = new FileReader()

      reader.onload = (e) => {
        const galleryItem = document.createElement("div")
        galleryItem.className = "gallery-item"

        const img = document.createElement("img")
        img.src = e.target.result
        img.alt = file.name

        const removeBtn = document.createElement("button")
        removeBtn.className = "remove-btn"
        removeBtn.innerHTML = '<i class="fas fa-times"></i>'
        removeBtn.addEventListener("click", () => removeFile(index))

        galleryItem.appendChild(img)
        galleryItem.appendChild(removeBtn)
        gallery.appendChild(galleryItem)
      }

      reader.readAsDataURL(file)
    })
  }

  function removeFile(index) {
    selectedFiles.splice(index, 1)
    updateFileCount()
    updateGallery()
    updateButtons()
  }

  function updateButtons() {
    if (selectedFiles.length > 0) {
      clearBtn.disabled = false
      analyzeBtn.disabled = false
    } else {
      clearBtn.disabled = true
      analyzeBtn.disabled = true
    }
  }

  clearBtn.addEventListener("click", () => {
    selectedFiles = []
    updateFileCount()
    updateGallery()
    updateButtons()
  })

  analyzeBtn.addEventListener("click", () => {
    // Show results section and hide upload section
    uploadSection.classList.add("hidden")
    resultsSection.classList.remove("hidden")

    // Show loading
    resultsLoading.classList.remove("hidden")
    resultsGrid.innerHTML = ""

    // Create form data
    const formData = new FormData()
    selectedFiles.forEach((file) => {
      formData.append("files[]", file)
    })

    // Send request to server
    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Hide loading
        resultsLoading.classList.add("hidden")

        // Store results
        results = data.results

        // Update summary
        totalImages.textContent = results.length

        // Count high risk results
        const highRiskCount = results.filter(
          (result) => result.risk_level === "High" || result.risk_level === "Very High",
        ).length
        highRisk.textContent = highRiskCount

        // Calculate average processing time
        const avgProcessingTime =
          results.reduce((sum, result) => sum + (result.processing_time || 0), 0) / results.length
        avgTime.textContent = avgProcessingTime.toFixed(2) + "s"

        // Display results
        displayResults(results)

        // Automatically generate and show summary report in chatbot
        generateSummaryReport(results)
      })
      .catch((error) => {
        console.error("Error:", error)
        resultsLoading.classList.add("hidden")
        alert("An error occurred while processing the images. Please try again.")
      })
  })

  // Fix the risk level display in the results card
  function displayResults(resultsArray) {
    resultsGrid.innerHTML = ""

    resultsArray.forEach((result) => {
      if (result.error) {
        // Handle error case
        const errorCard = document.createElement("div")
        errorCard.className = "result-card"
        errorCard.innerHTML = `
                    <div class="result-info">
                        <h3>Error Processing Image</h3>
                        <p>${result.original_filename}</p>
                        <p class="error-message">${result.error}</p>
                    </div>
                `
        resultsGrid.appendChild(errorCard)
        return
      }

      // Create result card
      const resultCard = document.createElement("div")
      resultCard.className = "result-card"
      resultCard.dataset.result = JSON.stringify(result)

      // Determine risk class - fix to handle both "Moderate" and "Medium" risk levels
      let riskClass = "low"
      if (result.risk_level === "Medium" || result.risk_level === "Moderate") riskClass = "medium"
      if (result.risk_level === "High") riskClass = "high"
      if (result.risk_level === "Very High") riskClass = "very-high"

      resultCard.innerHTML = `
                <div class="result-image">
                    <img src="/uploads/${result.filename}" alt="${result.original_filename}">
                </div>
                <div class="result-info">
                    <h3>${result.predicted_class}</h3>
                    <p>${result.original_filename}</p>
                    <div class="result-meta">
                        <span class="risk-badge ${riskClass}">${result.risk_level}</span>
                        <span class="confidence">${Math.round(result.confidence * 100)}% confidence</span>
                    </div>
                </div>
            `

      // Add click event to show details
      resultCard.addEventListener("click", () => showDetails(result))

      resultsGrid.appendChild(resultCard)
    })

    // Make the view summary button visible
    const viewSummaryBtn = document.getElementById("view-summary-btn")
    if (viewSummaryBtn) {
      viewSummaryBtn.style.display = "flex"
    }
  }

  // Fix the showDetails function to handle Medium risk level
  function showDetails(result) {
    // Store current result for chatbot context
    currentResult = result

    // Populate modal with result data
    document.getElementById("detail-img").src = `/uploads/${result.filename}`
    document.getElementById("detail-diagnosis").textContent = result.predicted_class
    document.getElementById("confidence-value").textContent = `${Math.round(result.confidence * 100)}%`
    document.getElementById("confidence-bar").style.width = `${result.confidence * 100}%`

    const riskElement = document.getElementById("risk-level")
    riskElement.textContent = result.risk_level
    riskElement.className = "risk-badge"

    // Add risk class - fix to handle both "Moderate" and "Medium" risk levels
    if (result.risk_level === "Low") riskElement.classList.add("low")
    if (result.risk_level === "Medium" || result.risk_level === "Moderate") riskElement.classList.add("medium")
    if (result.risk_level === "High") riskElement.classList.add("high")
    if (result.risk_level === "Very High") riskElement.classList.add("very-high")

    document.getElementById("description-text").textContent = result.description
    document.getElementById("detail-chart").src = `data:image/png;base64,${result.visualization}`

    // Show modal
    detailModal.style.display = "block"

    // Set context for chatbot
    if (window.setChatbotContext) {
      window.setChatbotContext(result)
    }
  }

  // Fix the generateSummaryReport function to handle Medium risk level
  function generateSummaryReport(results) {
    if (results.length === 0) return

    // Count by diagnosis
    const diagnosisCounts = {}
    results.forEach((result) => {
      if (!result.error) {
        const diagnosis = result.predicted_class
        diagnosisCounts[diagnosis] = (diagnosisCounts[diagnosis] || 0) + 1
      }
    })

    // Count by risk level
    const riskCounts = {
      Low: 0,
      Medium: 0,
      Moderate: 0, // Include both Medium and Moderate for compatibility
      High: 0,
      "Very High": 0,
    }

    results.forEach((result) => {
      if (!result.error) {
        // Handle both "Medium" and "Moderate" as the same category
        if (result.risk_level === "Moderate") {
          riskCounts["Medium"] = (riskCounts["Medium"] || 0) + 1
        } else {
          riskCounts[result.risk_level] = (riskCounts[result.risk_level] || 0) + 1
        }
      }
    })

    // Combine Medium and Moderate counts for display
    const mediumCount = (riskCounts["Medium"] || 0) + (riskCounts["Moderate"] || 0)

    // Find highest confidence result
    let highestConfidence = 0
    let highestConfidenceResult = null

    results.forEach((result) => {
      if (!result.error && result.confidence > highestConfidence) {
        highestConfidence = result.confidence
        highestConfidenceResult = result
      }
    })

    // Find highest risk results
    const highRiskResults = results.filter(
      (result) => !result.error && (result.risk_level === "High" || result.risk_level === "Very High"),
    )

    // Generate summary HTML
    let summaryHTML = `
      <div class="summary-report">
        <h2>Analysis Summary Report</h2>
        
        <p>We've analyzed ${results.length} image${results.length !== 1 ? "s" : ""} and here's a comprehensive summary of our findings:</p>
        
        <div class="risk-summary">
          <div class="risk-item">
            <h4>Total Images</h4>
            <p>${results.length}</p>
          </div>
          <div class="risk-item">
            <h4>High Risk Findings</h4>
            <p>${riskCounts["High"] + riskCounts["Very High"]}</p>
          </div>
          <div class="risk-item">
            <h4>Medium Risk</h4>
            <p>${mediumCount}</p>
          </div>
          <div class="risk-item">
            <h4>Low Risk</h4>
            <p>${riskCounts["Low"]}</p>
          </div>
        </div>

        <h3>Risk Level Breakdown</h3>
        <ul>
    `

    if (riskCounts["Very High"] > 0) {
      summaryHTML += `<li><strong>Very High Risk:</strong> ${riskCounts["Very High"]} image${riskCounts["Very High"] !== 1 ? "s" : ""}</li>`
    }
    if (riskCounts["High"] > 0) {
      summaryHTML += `<li><strong>High Risk:</strong> ${riskCounts["High"]} image${riskCounts["High"] !== 1 ? "s" : ""}</li>`
    }
    if (mediumCount > 0) {
      summaryHTML += `<li><strong>Medium Risk:</strong> ${mediumCount} image${mediumCount !== 1 ? "s" : ""}</li>`
    }
    if (riskCounts["Low"] > 0) {
      summaryHTML += `<li><strong>Low Risk:</strong> ${riskCounts["Low"]} image${riskCounts["Low"] !== 1 ? "s" : ""}</li>`
    }

    summaryHTML += `
        </ul>
        
        <h3>Diagnosis Breakdown</h3>
        <ul>
    `

    Object.keys(diagnosisCounts).forEach((diagnosis) => {
      summaryHTML += `<li><strong>${diagnosis}:</strong> ${diagnosisCounts[diagnosis]} image${diagnosisCounts[diagnosis] !== 1 ? "s" : ""}</li>`
    })

    summaryHTML += `
        </ul>
    `

    // Add high risk findings section if applicable
    if (highRiskResults.length > 0) {
      summaryHTML += `
        <h3>High Risk Findings</h3>
        <p>The following high-risk conditions were detected and require attention:</p>
        <ul>
      `

      highRiskResults.forEach((result) => {
        summaryHTML += `
          <li>
            <strong>${result.predicted_class}</strong> (${Math.round(result.confidence * 100)}% confidence) - 
            ${result.description}
          </li>
        `
      })

      summaryHTML += `
        </ul>
      `
    }

    // Most confident diagnosis
    if (highestConfidenceResult) {
      summaryHTML += `
        <h3>Most Confident Diagnosis</h3>
        <p><strong>${highestConfidenceResult.predicted_class}</strong> (${Math.round(highestConfidenceResult.confidence * 100)}% confidence)</p>
        <p><strong>Risk Level:</strong> ${highestConfidenceResult.risk_level}</p>
        <p><strong>Description:</strong> ${highestConfidenceResult.description}</p>
      `
    }

    // Add recommendations section
    summaryHTML += `
      <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
    `

    if (riskCounts["Very High"] > 0) {
      summaryHTML += `
        <li><strong>URGENT:</strong> Please consult with a dermatologist as soon as possible about the very high-risk findings. These conditions can be serious and require prompt medical attention.</li>
      `
    } else if (riskCounts["High"] > 0) {
      summaryHTML += `
        <li><strong>IMPORTANT:</strong> Schedule an appointment with a dermatologist within the next few weeks to evaluate the high-risk findings. Early treatment is key for better outcomes.</li>
      `
    } else if (mediumCount > 0) {
      summaryHTML += `
        <li><strong>RECOMMENDED:</strong> Schedule a routine appointment with a dermatologist to evaluate the moderate-risk findings. While not urgent, these should be professionally assessed.</li>
      `
    } else {
      summaryHTML += `
        <li><strong>ROUTINE:</strong> Consider discussing these findings during your next regular check-up. Low-risk findings generally don't require immediate attention but should be monitored.</li>
      `
    }

    summaryHTML += `
        <li><strong>MONITORING:</strong> Continue to monitor your skin for any changes using the ABCDE method:
          <ul>
            <li><strong>A</strong>symmetry: One half doesn't match the other</li>
            <li><strong>B</strong>order irregularity: Edges are ragged or blurred</li>
            <li><strong>C</strong>olor variations: Multiple colors within the same lesion</li>
            <li><strong>D</strong>iameter: Larger than 6mm (pencil eraser size)</li>
            <li><strong>E</strong>volving: Changes in size, shape, color, or symptoms</li>
          </ul>
        </li>
        <li><strong>PREVENTION:</strong> Practice sun safety with SPF 30+ sunscreen, protective clothing, and limiting exposure during peak hours (10am-4pm).</li>
        <li><strong>FOLLOW-UP:</strong> Consider annual skin checks with a dermatologist, especially if you have risk factors like fair skin, history of sunburns, or family history of skin cancer.</li>
      </ul>
      </div>
      
      <div class="disclaimer">
        <p>IMPORTANT: This is an AI-generated analysis and should not replace professional medical advice. Always consult with a qualified healthcare provider for proper evaluation and treatment of skin conditions.</p>
      </div>
    </div>
    `

    // Open modal and display summary
    const chatbotModal = document.getElementById("chatbot-modal")
    if (chatbotModal) {
      chatbotModal.style.display = "block"

      const chatbotMessages = document.getElementById("chatbot-messages")
      chatbotMessages.innerHTML = summaryHTML
    }
  }

  backBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden")
    uploadSection.classList.remove("hidden")
  })

  // Search and filter functionality
  searchResults.addEventListener("input", filterResultsDisplay)
  filterResults.addEventListener("change", filterResultsDisplay)

  function filterResultsDisplay() {
    const searchTerm = searchResults.value.toLowerCase()
    const filterValue = filterResults.value

    let filteredResults = [...results]

    // Apply search filter
    if (searchTerm) {
      filteredResults = filteredResults.filter(
        (result) =>
          result.original_filename.toLowerCase().includes(searchTerm) ||
          result.predicted_class.toLowerCase().includes(searchTerm),
      )
    }

    // Apply risk filter
    if (filterValue === "high-risk") {
      filteredResults = filteredResults.filter(
        (result) => result.risk_level === "High" || result.risk_level === "Very High",
      )
    } else if (filterValue === "low-risk") {
      filteredResults = filteredResults.filter(
        (result) => result.risk_level === "Low" || result.risk_level === "Moderate",
      )
    }

    displayResults(filteredResults)
  }

  // Export results functionality
  exportBtn.addEventListener("click", () => {
    // Create CSV content
    let csvContent = "data:text/csv;charset=utf-8,"
    csvContent += "Filename,Diagnosis,Confidence,Risk Level\n"

    results.forEach((result) => {
      if (!result.error) {
        csvContent += `${result.original_filename},${result.predicted_class},${Math.round(result.confidence * 100)}%,${result.risk_level}\n`
      }
    })

    // Create download link
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "skin_analysis_results.csv")
    document.body.appendChild(link)

    // Trigger download
    link.click()
    document.body.removeChild(link)
  })

  // Function to open chatbot with context
  function openChatbot(result) {
    const chatbotModal = document.getElementById("chatbot-modal")
    if (chatbotModal) {
      chatbotModal.style.display = "block"

      // If we have a result, add a welcome message with context
      if (result) {
        const chatbotMessages = document.getElementById("chatbot-messages")
        chatbotMessages.innerHTML = `
                    <div class="message bot-message">
                        <div class="message-content">
                            <p>Hello! I'm your SkinScan AI Assistant. I can help explain your diagnosis of ${result.predicted_class} and answer any questions you might have about it. What would you like to know?</p>
                        </div>
                    </div>
                `
      }
    }
  }

  // Function to generate and display summary report in chatbot
  // Add the view summary button event listener
  const viewSummaryBtn = document.getElementById("view-summary-btn")
  if (viewSummaryBtn) {
    viewSummaryBtn.addEventListener("click", () => {
      generateSummaryReport(results)
    })
  }

  // Expose the function to the global scope
  window.openChatbot = openChatbot
  window.generateSummaryReport = generateSummaryReport
})
