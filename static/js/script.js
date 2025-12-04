// Elementos DOM
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const loading = document.getElementById('loading');

// Variáveis globais
let currentFile = null;

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#764ba2';
        uploadArea.style.background = '#f8f9ff';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'white';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'white';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
});

// Manipulação de arquivo
function handleFile(file) {
    if (!file.type.match('image.*')) {
        alert('Por favor, selecione apenas arquivos de imagem.');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        alert('O arquivo é muito grande. Tamanho máximo: 16MB');
        return;
    }
    
    currentFile = file;
    
    // Mostrar pré-visualização
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        fileName.textContent = `Nome: ${file.name}`;
        fileSize.textContent = `Tamanho: ${(file.size / 1024).toFixed(2)} KB`;
        
        // Rolar para a pré-visualização
        previewSection.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(file);
}

// Upload e busca
async function searchSimilar() {
    if (!currentFile) {
        alert('Por favor, selecione uma imagem primeiro.');
        return;
    }
    
    // Mostrar loading
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Criar FormData
    const formData = new FormData();
    formData.append('file', currentFile);
    
    try {
        // Fazer upload
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const uploadData = await uploadResponse.json();
        
        if (!uploadData.success) {
            throw new Error(uploadData.error || 'Erro no upload');
        }
        
        // Buscar imagens similares
        const searchResponse = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: uploadData.filename,
                k: 5
            })
        });
        
        const searchData = await searchResponse.json();
        
        if (!searchData.success) {
            throw new Error(searchData.error || 'Erro na busca');
        }
        
        // Mostrar resultados
        displayResults(searchData.results);
        
    } catch (error) {
        console.error('Erro:', error);
        alert(`Erro: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

// Exibir resultados
function displayResults(results) {
    resultsGrid.innerHTML = '';
    
    if (results.length === 0) {
        resultsGrid.innerHTML = `
            <div class="no-results">
                <i class="fas fa-search-minus fa-3x"></i>
                <h3>Nenhuma imagem similar encontrada</h3>
                <p>Tente com outra imagem</p>
            </div>
        `;
    } else {
        results.forEach(result => {
            const similarityPercent = (100 - result.distance * 10).toFixed(1);
            
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.innerHTML = `
                <div class="similarity-badge">
                    <i class="fas fa-percentage"></i> ${similarityPercent}%
                </div>
                <img src="${result.path}" 
                     alt="Imagem similar" 
                     onerror="this.src='https://via.placeholder.com/300x200?text=Imagem+não+encontrada'">
                <div class="result-info">
                    <h4>${result.filename}</h4>
                    <p><i class="fas fa-ruler"></i> Distância: ${result.distance.toFixed(4)}</p>
                </div>
            `;
            
            resultsGrid.appendChild(resultCard);
        });
    }
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Limpar upload
function clearUpload() {
    currentFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Modal functions
function showAbout() {
    document.getElementById('aboutModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('aboutModal').style.display = 'none';
}

// Fechar modal clicando fora
window.onclick = function(event) {
    const modal = document.getElementById('aboutModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}