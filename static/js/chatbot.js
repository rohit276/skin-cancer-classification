document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const openChatBtn = document.getElementById("open-chat-btn")
  const chatbotModal = document.getElementById("chatbot-modal")
  const closeChatbot = document.querySelector(".close-chatbot")
  const chatMessages = document.getElementById("chatbot-messages")
  const chatInput = document.getElementById("chat-input")
  const sendChatBtn = document.getElementById("send-chat-btn")
  const suggestionBtns = document.querySelectorAll(".suggestion-btn")

  // Variables
  const currentContext = {}

  // Event Listeners
  if (openChatBtn) {
    openChatBtn.addEventListener("click", () => {
      // When clicking the summary report button in the nav, generate a report if we have results
      if (window.results && window.results.length > 0) {
        window.generateSummaryReport(window.results)
      } else {
        chatbotModal.style.display = "block"
        chatMessages.innerHTML = `
          <div class="summary-report">
            <h2>No Analysis Results Available</h2>
            <p>Please upload and analyze images first to generate a summary report.</p>
          </div>
        `
      }
    })
  }

  if (closeChatbot) {
    closeChatbot.addEventListener("click", () => {
      chatbotModal.style.display = "none"
    })
  }

  // Close modal when clicking outside the content
  window.addEventListener("click", (e) => {
    if (e.target === chatbotModal) {
      chatbotModal.style.display = "none"
    }
  })

  // Send message on button click
  if (sendChatBtn) {
    sendChatBtn.addEventListener("click", sendMessage)
  }

  // Send message on Enter key
  if (chatInput) {
    chatInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        sendMessage()
      }
    })
  }

  // Suggestion buttons
  suggestionBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const query = btn.dataset.query
      if (query) {
        chatInput.value = query
        sendMessage()
      }
    })
  })

  // Functions
  function sendMessage() {
    const message = chatInput.value.trim()

    if (message === "") return

    // Add user message to chat
    addMessage(message, "user")

    // Clear input
    chatInput.value = ""

    // Show typing indicator
    showTypingIndicator()

    // Send to server
    fetch("/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: message,
        context: currentContext,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Remove typing indicator
        removeTypingIndicator()

        // Add bot response
        addMessage(data.response, "bot")

        // Scroll to bottom
        scrollToBottom()
      })
      .catch((error) => {
        console.error("Error:", error)
        removeTypingIndicator()
        addMessage("I'm sorry, I encountered an error processing your request. Please try again.", "bot")
        scrollToBottom()
      })
  }

  function addMessage(text, sender) {
    const messageDiv = document.createElement("div")
    messageDiv.className = `message ${sender}-message`

    const contentDiv = document.createElement("div")
    contentDiv.className = "message-content"

    // Check if the text contains HTML tags (for formatted summary reports)
    if (text.includes("<h2>") || text.includes("<h3>") || text.includes("<strong>")) {
      contentDiv.innerHTML = text
    } else {
      const paragraph = document.createElement("p")
      paragraph.textContent = text
      contentDiv.appendChild(paragraph)
    }

    messageDiv.appendChild(contentDiv)
    chatMessages.appendChild(messageDiv)

    scrollToBottom()
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement("div")
    typingDiv.className = "message bot-message typing-indicator"
    typingDiv.id = "typing-indicator"

    const contentDiv = document.createElement("div")
    contentDiv.className = "message-content"

    contentDiv.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>'

    typingDiv.appendChild(contentDiv)
    chatMessages.appendChild(typingDiv)

    scrollToBottom()
  }

  function removeTypingIndicator() {
    const typingIndicator = document.getElementById("typing-indicator")
    if (typingIndicator) {
      typingIndicator.remove()
    }
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight
  }

  // Set context from result
  window.setChatbotContext = (result) => {
    if (result) {
      window.currentContext = {
        diagnosis: result.predicted_class,
        confidence: result.confidence,
        risk_level: result.risk_level,
      }
    }
  }
})

