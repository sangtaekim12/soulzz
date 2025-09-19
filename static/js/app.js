// 전역 변수
let isSystemReady = false;
let isWaitingForResponse = false;

// DOM 요소들
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const statusAlert = document.getElementById('status-alert');
const exampleBtns = document.querySelectorAll('.example-btn');
const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

// 초기화
document.addEventListener('DOMContentLoaded', function() {
    checkSystemStatus();
    setupEventListeners();
});

// 이벤트 리스너 설정
function setupEventListeners() {
    // 전송 버튼 클릭
    sendButton.addEventListener('click', sendMessage);
    
    // Enter 키 입력
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 예시 질문 버튼들
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            if (!btn.disabled) {
                userInput.value = btn.textContent.trim();
                sendMessage();
            }
        });
    });
}

// 시스템 상태 확인
function checkSystemStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            updateStatusAlert(data);
            
            if (data.server_ready && data.initialized) {
                isSystemReady = true;
                enableInterface();
            } else {
                isSystemReady = false;
                // 5초 후 다시 확인
                setTimeout(checkSystemStatus, 5000);
            }
        })
        .catch(error => {
            console.error('Status check failed:', error);
            updateStatusAlert({
                server_ready: false,
                error: '서버 연결에 실패했습니다.'
            });
            // 10초 후 다시 확인
            setTimeout(checkSystemStatus, 10000);
        });
}

// 상태 알림 업데이트
function updateStatusAlert(status) {
    const alertDiv = statusAlert;
    
    if (status.server_ready && status.initialized) {
        alertDiv.className = 'alert alert-success d-flex align-items-center mb-4';
        alertDiv.innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            <span>시스템이 준비되었습니다! (문서 ${status.documents_loaded}개 로드됨)</span>
        `;
    } else if (status.error) {
        alertDiv.className = 'alert alert-warning d-flex align-items-center mb-4';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            <span>${status.error}</span>
        `;
    } else {
        alertDiv.className = 'alert alert-info d-flex align-items-center mb-4';
        alertDiv.innerHTML = `
            <div class="spinner-border spinner-border-sm me-2" role="status">
                <span class="visually-hidden">로딩 중...</span>
            </div>
            <span>시스템을 초기화하는 중입니다...</span>
        `;
    }
}

// 인터페이스 활성화
function enableInterface() {
    userInput.disabled = false;
    sendButton.disabled = false;
    exampleBtns.forEach(btn => {
        btn.disabled = false;
    });
    userInput.focus();
}

// 메시지 전송
function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message || !isSystemReady || isWaitingForResponse) {
        return;
    }
    
    // 사용자 메시지 추가
    addMessage(message, 'user');
    userInput.value = '';
    
    // 입력 비활성화
    isWaitingForResponse = true;
    userInput.disabled = true;
    sendButton.disabled = true;
    
    // 타이핑 인디케이터 추가
    const typingId = addTypingIndicator();
    
    // 서버에 요청 전송
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // 타이핑 인디케이터 제거
        removeTypingIndicator(typingId);
        
        if (data.error) {
            addMessage(data.error, 'bot', []);
        } else {
            addMessage(data.answer, 'bot', data.sources);
        }
    })
    .catch(error => {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addMessage('죄송합니다. 네트워크 오류가 발생했습니다. 다시 시도해주세요.', 'bot', []);
    })
    .finally(() => {
        // 입력 다시 활성화
        isWaitingForResponse = false;
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    });
}

// 메시지 추가
function addMessage(text, sender, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;
    
    content.appendChild(textDiv);
    
    // 출처 정보 추가 (봇 메시지인 경우)
    if (sender === 'bot' && sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.textContent = source;
            sourcesDiv.appendChild(sourceItem);
        });
        
        content.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// 타이핑 인디케이터 추가
function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'message bot-message';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
    
    return typingId;
}

// 타이핑 인디케이터 제거
function removeTypingIndicator(typingId) {
    const typingElement = document.getElementById(typingId);
    if (typingElement) {
        typingElement.remove();
    }
}

// 채팅 영역을 아래로 스크롤
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 페이지 가시성 변경 시 상태 확인
document.addEventListener('visibilitychange', function() {
    if (!document.hidden && !isSystemReady) {
        checkSystemStatus();
    }
});

// 네트워크 상태 변경 감지
window.addEventListener('online', function() {
    if (!isSystemReady) {
        checkSystemStatus();
    }
});

// 에러 발생 시 전역 핸들링
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
});

// 미처리 Promise 거부 핸들링
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});