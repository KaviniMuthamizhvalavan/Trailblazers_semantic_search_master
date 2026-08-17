/**
 * Trailblazers Semantic Search & Recommendation Engine
 * Frontend JavaScript Controller
 */

// Global API Base URL (auto-detects current host or defaults to http://127.0.0.1:8000)
const API_BASE = window.location.origin.startsWith('http') 
  ? window.location.origin 
  : 'http://127.0.0.1:8000';

// State Store
const state = {
  products: [],
  searchResults: [],
  currentQuery: '',
  isSearching: false,
  indexStatus: null
};

// DOM Element Selectors
const elements = {
  searchInput: document.getElementById('search-input'),
  searchSubmitBtn: document.getElementById('search-submit-btn'),
  searchClearBtn: document.getElementById('search-clear-btn'),
  productGrid: document.getElementById('product-grid'),
  sectionHeading: document.getElementById('section-heading'),
  resultsCountBadge: document.getElementById('results-count-badge'),
  resetCatalogBtn: document.getElementById('reset-catalog-btn'),
  emptyState: document.getElementById('empty-state'),
  loadingState: document.getElementById('loading-state'),
  loadingMessage: document.getElementById('loading-message'),
  statusText: document.getElementById('status-text'),
  statusDim: document.getElementById('status-dim'),
  
  // Modal Elements
  recommendationModal: document.getElementById('recommendation-modal'),
  modalCloseBtn: document.getElementById('modal-close-btn'),
  modalProductTitle: document.getElementById('modal-product-title'),
  modalProductDesc: document.getElementById('modal-product-desc'),
  modalProductPrice: document.getElementById('modal-product-price'),
  modalProductRating: document.getElementById('modal-product-rating'),
  modalProductStock: document.getElementById('modal-product-stock'),
  modalProductCategory: document.getElementById('modal-product-category'),
  modalRecGrid: document.getElementById('modal-rec-grid'),
  modalRecLoading: document.getElementById('modal-rec-loading'),
  
  toastContainer: document.getElementById('toast-container')
};

// Initialization on DOM Load
document.addEventListener('DOMContentLoaded', () => {
  initApp();
  setupEventListeners();
});

async function initApp() {
  await fetchIndexStatus();
  await loadFullCatalog();
}

/**
 * Fetch Health & Index Status from /index/status
 */
async function fetchIndexStatus() {
  try {
    const response = await fetch(`${API_BASE}/index/status`);
    if (response.ok) {
      const data = await response.json();
      state.indexStatus = data;
      elements.statusText.textContent = `FAISS Index: Ready (${data.total_products} vectors)`;
      elements.statusDim.textContent = `${data.index_dimension}-dim`;
    } else {
      elements.statusText.textContent = 'FAISS Index: Error';
    }
  } catch (err) {
    console.warn('Index status check failed:', err);
    elements.statusText.textContent = 'Backend Connected';
  }
}

/**
 * Fetch Full Catalog from /products
 */
async function loadFullCatalog() {
  showLoading('Loading product catalog...');
  try {
    const response = await fetch(`${API_BASE}/products`);
    if (!response.ok) throw new Error('Failed to load products');
    
    const data = await response.json();
    state.products = data.products || [];
    state.currentQuery = '';
    
    elements.sectionHeading.textContent = 'Complete Product Catalog';
    elements.resultsCountBadge.textContent = `${state.products.length} items`;
    elements.searchInput.value = '';
    elements.searchClearBtn.classList.add('hidden');
    
    renderProductCards(state.products, false);
    hideLoading();
  } catch (err) {
    hideLoading();
    showToast(`Error connecting to API server: ${err.message}`, 'error');
    showEmptyState('Could not reach backend API server. Make sure FastAPI server is running on http://127.0.0.1:8000.');
  }
}

/**
 * Perform Natural Language Semantic Search via /search?q=...
 */
async function performSearch(query) {
  const trimmed = query ? query.trim() : '';
  if (!trimmed) {
    loadFullCatalog();
    return;
  }

  state.currentQuery = trimmed;
  showLoading(`Embedding query "${trimmed}" & querying FAISS vector space...`);
  
  try {
    const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(trimmed)}&top_n=20`);
    if (!response.ok) throw new Error('Search failed');
    
    const data = await response.json();
    state.searchResults = data.results || [];
    
    elements.sectionHeading.innerHTML = `Semantic Results for <span class="accent-text">"${trimmed}"</span>`;
    elements.resultsCountBadge.textContent = `${state.searchResults.length} matches`;
    
    hideLoading();
    
    if (state.searchResults.length === 0) {
      showEmptyState(`No semantic matches found for "${trimmed}".`);
    } else {
      renderProductCards(state.searchResults, true);
    }
  } catch (err) {
    hideLoading();
    showToast(`Search error: ${err.message}`, 'error');
  }
}

/**
 * Render Product Grid Cards
 * @param {Array} items 
 * @param {Boolean} isSearchResult 
 */
function renderProductCards(items, isSearchResult = false) {
  elements.emptyState.classList.add('hidden');
  elements.productGrid.innerHTML = '';

  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'product-card';

    // Format Rating Stars
    const ratingNum = Number(item.rating || 0).toFixed(1);
    const inStock = item.in_stock !== false;
    
    // Scores markup (if search result)
    let scoresHTML = '';
    if (isSearchResult && item.similarity_score !== undefined) {
      const simPercent = Math.round((item.similarity_score || 0) * 100);
      const rerankVal = Number(item.reranked_score || item.similarity_score || 0).toFixed(3);
      
      scoresHTML = `
        <div class="scores-section">
          <span class="score-chip" title="Vector Cosine Similarity">
            <i class="fa-solid fa-microchip"></i> Vector Sim: ${simPercent}%
          </span>
          <span class="score-chip score-chip-rerank" title="Re-ranked score incorporating ratings & stock">
            <i class="fa-solid fa-chart-line"></i> Re-rank: ${rerankVal}
          </span>
        </div>
      `;
    }

    card.innerHTML = `
      <div>
        <div class="card-top">
          <span class="badge badge-category">${escapeHTML(item.category || 'General')}</span>
          <span class="badge ${inStock ? 'badge-stock' : 'badge-out-of-stock'}">
            <i class="fa-solid ${inStock ? 'fa-check' : 'fa-xmark'}"></i> ${inStock ? 'In Stock' : 'Out of Stock'}
          </span>
        </div>
        
        <h3 class="card-title">${escapeHTML(item.title)}</h3>
        <p class="card-desc">${escapeHTML(item.description)}</p>
      </div>

      <div>
        ${scoresHTML}
        
        <div class="card-footer">
          <div>
            <div class="product-price">$${Number(item.price).toFixed(2)}</div>
            <span class="badge badge-rating" style="margin-top: 4px;">
              <i class="fa-solid fa-star"></i> ${ratingNum}
            </span>
          </div>

          <button class="btn-similar" onclick="openRecommendations('${item.product_id}')">
            <i class="fa-solid fa-diagram-project"></i> Similar
          </button>
        </div>
      </div>
    `;

    elements.productGrid.appendChild(card);
  });
}

/**
 * Fetch and Open Recommendation Modal for a Product ID
 */
async function openRecommendations(productId) {
  elements.recommendationModal.classList.remove('hidden');
  elements.modalRecLoading.classList.remove('hidden');
  elements.modalRecGrid.innerHTML = '';
  document.body.style.overflow = 'hidden';

  try {
    const response = await fetch(`${API_BASE}/products/${productId}/similar?top_n=4`);
    if (!response.ok) throw new Error('Could not fetch recommendations');

    const data = await response.json();
    const source = data.source_product;
    const recs = data.similar_products || [];

    // Populate Source Product Details
    elements.modalProductTitle.textContent = source.title;
    elements.modalProductDesc.textContent = source.description;
    elements.modalProductPrice.textContent = `$${Number(source.price).toFixed(2)}`;
    elements.modalProductRating.innerHTML = `<i class="fa-solid fa-star"></i> ${Number(source.rating).toFixed(1)}`;
    elements.modalProductCategory.textContent = source.category || 'General';
    
    const inStock = source.in_stock !== false;
    elements.modalProductStock.className = `badge ${inStock ? 'badge-stock' : 'badge-out-of-stock'}`;
    elements.modalProductStock.textContent = inStock ? 'In Stock' : 'Out of Stock';

    elements.modalRecLoading.classList.add('hidden');

    // Render Similar Products inside Modal
    if (recs.length === 0) {
      elements.modalRecGrid.innerHTML = '<p class="text-muted">No similar products found in vector space.</p>';
      return;
    }

    recs.forEach(rec => {
      const recCard = document.createElement('div');
      recCard.className = 'product-card';
      recCard.style.padding = '1rem';
      
      const simPercent = Math.round((rec.similarity_score || 0) * 100);

      recCard.innerHTML = `
        <div>
          <div class="card-top">
            <span class="badge badge-category">${escapeHTML(rec.category || 'Item')}</span>
            <span class="score-chip" style="font-size:0.7rem;">
              Sim: ${simPercent}%
            </span>
          </div>
          <h4 class="card-title" style="font-size:1rem;">${escapeHTML(rec.title)}</h4>
          <p class="card-desc" style="-webkit-line-clamp:2; font-size:0.82rem;">${escapeHTML(rec.description)}</p>
        </div>
        <div class="card-footer" style="padding-top:0.5rem;">
          <span class="product-price" style="font-size:1.05rem;">$${Number(rec.price).toFixed(2)}</span>
          <button class="btn-similar" style="padding:0.3rem 0.6rem; font-size:0.75rem;" onclick="openRecommendations('${rec.product_id}')">
            View
          </button>
        </div>
      `;

      elements.modalRecGrid.appendChild(recCard);
    });

  } catch (err) {
    elements.modalRecLoading.classList.add('hidden');
    showToast(`Failed to fetch recommendations: ${err.message}`, 'error');
  }
}

function closeModal() {
  elements.recommendationModal.classList.add('hidden');
  document.body.style.overflow = '';
}

/**
 * Event Listeners Setup
 */
function setupEventListeners() {
  // Search Button Click
  elements.searchSubmitBtn.addEventListener('click', () => {
    performSearch(elements.searchInput.value);
  });

  // Enter key inside search box
  elements.searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      performSearch(elements.searchInput.value);
    }
  });

  // Input change toggle clear button
  elements.searchInput.addEventListener('input', (e) => {
    if (e.target.value.trim().length > 0) {
      elements.searchClearBtn.classList.remove('hidden');
    } else {
      elements.searchClearBtn.classList.add('hidden');
    }
  });

  // Clear button click
  elements.searchClearBtn.addEventListener('click', () => {
    elements.searchInput.value = '';
    elements.searchClearBtn.classList.add('hidden');
    loadFullCatalog();
  });

  // Reset / Show All Catalog Button
  elements.resetCatalogBtn.addEventListener('click', () => {
    loadFullCatalog();
  });

  // Quick Query Tags
  document.querySelectorAll('.tag-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-query');
      elements.searchInput.value = q;
      elements.searchClearBtn.classList.remove('hidden');
      performSearch(q);
    });
  });

  // Modal Close Listeners
  elements.modalCloseBtn.addEventListener('click', closeModal);
  elements.recommendationModal.addEventListener('click', (e) => {
    if (e.target === elements.recommendationModal) closeModal();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !elements.recommendationModal.classList.contains('hidden')) {
      closeModal();
    }
  });
}

/**
 * Helpers
 */
function showLoading(msg = 'Searching...') {
  elements.loadingMessage.textContent = msg;
  elements.loadingState.classList.remove('hidden');
  elements.productGrid.classList.add('hidden');
  elements.emptyState.classList.add('hidden');
}

function hideLoading() {
  elements.loadingState.classList.add('hidden');
  elements.productGrid.classList.remove('hidden');
}

function showEmptyState(msg) {
  elements.productGrid.innerHTML = '';
  elements.emptyState.querySelector('p').textContent = msg;
  elements.emptyState.classList.remove('hidden');
}

function resetToCatalog() {
  loadFullCatalog();
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <i class="fa-solid ${type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-check'}"></i>
    <span>${escapeHTML(message)}</span>
  `;
  
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 4000);
}

function escapeHTML(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
