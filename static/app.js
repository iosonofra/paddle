/**
 * PaddleOCR Studio - Interactive Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const canvasContainer = document.getElementById("canvasContainer");
  const canvasWrapper = document.getElementById("canvasWrapper");
  const canvas = document.getElementById("mainCanvas");
  const ctx = canvas.getContext("2d");
  const viewport = document.getElementById("viewport");
  
  // Controls & Settings
  const langSelect = document.getElementById("langSelect");
  const angleClsToggle = document.getElementById("angleClsToggle");
  const hwStatus = document.getElementById("hwStatus");
  const hwPill = document.getElementById("hwPill");
  
  // Toolbar Buttons
  const zoomInBtn = document.getElementById("zoomInBtn");
  const zoomOutBtn = document.getElementById("zoomOutBtn");
  const zoomFitBtn = document.getElementById("zoomFitBtn");
  const zoomDisplay = document.getElementById("zoomDisplay");
  const toggleBoxesBtn = document.getElementById("toggleBoxesBtn");
  const clearBtn = document.getElementById("clearBtn");
  const pdfNav = document.getElementById("pdfNav");
  const prevPageBtn = document.getElementById("prevPageBtn");
  const nextPageBtn = document.getElementById("nextPageBtn");
  const pageIndicator = document.getElementById("pageIndicator");

  // Loading Overlay
  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");

  // Stats
  const statBoxes = document.getElementById("statBoxes");
  const statConf = document.getElementById("statConf");
  const statTime = document.getElementById("statTime");
  const statDevice = document.getElementById("statDevice");

  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const fullTextOutput = document.getElementById("fullTextOutput");
  const linesListContainer = document.getElementById("linesListContainer");
  const jsonOutput = document.getElementById("jsonOutput");
  const searchInput = document.getElementById("searchInput");
  const clearSearchBtn = document.getElementById("clearSearchBtn");
  const searchMatchCount = document.getElementById("searchMatchCount");
  const searchNavBtns = document.getElementById("searchNavBtns");
  const searchPrevBtn = document.getElementById("searchPrevBtn");
  const searchNextBtn = document.getElementById("searchNextBtn");
  const searchTabBtn = document.getElementById("searchTabBtn");
  const searchBadgeCount = document.getElementById("searchBadgeCount");
  const searchResultsContainer = document.getElementById("searchResultsContainer");

  // Actions
  const copyTextBtn = document.getElementById("copyTextBtn");
  const printDocBtn = document.getElementById("printDocBtn");
  const printDropdownBtn = document.getElementById("printDropdownBtn");
  const exportDropdownBtn = document.getElementById("exportDropdownBtn");
  const exportMenu = document.getElementById("exportMenu");
  const exportPdfSearchableBtn = document.getElementById("exportPdfSearchableBtn");
  const exportPdfReconstructedBtn = document.getElementById("exportPdfReconstructedBtn");
  const exportTxtBtn = document.getElementById("exportTxtBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");
  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const toastContainer = document.getElementById("toastContainer");

  // State Variables
  let currentOcrData = null;
  let currentPageIndex = 0;
  let loadedImage = new Image();
  let showBoxes = true;
  let hoveredBoxId = null;
  let selectedBoxId = null;
  let searchQuery = "";
  let searchResults = [];
  let currentMatchIndex = -1;

  // Viewport Transform (Zoom & Pan)
  let scale = 1.0;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let startX = 0;
  let startY = 0;

  // Reset completo dello stato dell'applicazione e dei campi testo
  function resetAppState() {
    currentOcrData = null;
    currentPageIndex = 0;
    if (fileInput) fileInput.value = "";
    if (fullTextOutput) fullTextOutput.value = "";
    if (linesListContainer) linesListContainer.innerHTML = `<div class="empty-state">Nessun dato ancora elaborato.</div>`;
    if (jsonOutput) jsonOutput.textContent = "{}";
    if (searchInput) searchInput.value = "";
    if (searchMatchCount) searchMatchCount.classList.add("hidden");
    if (searchNavBtns) searchNavBtns.classList.add("hidden");
    if (clearSearchBtn) clearSearchBtn.classList.add("hidden");
    if (searchBadgeCount) {
      searchBadgeCount.classList.add("hidden");
      searchBadgeCount.textContent = "0";
    }
    if (searchResultsContainer) searchResultsContainer.innerHTML = "";
    if (statBoxes) statBoxes.textContent = "0";
    if (statConf) statConf.textContent = "0.0%";
    if (statTime) statTime.textContent = "0 ms";
    if (canvasContainer) canvasContainer.classList.add("hidden");
    if (dropzone) dropzone.classList.remove("hidden");
    if (pdfNav) pdfNav.classList.add("hidden");
  }

  // Pulisci subito l'interfaccia all'avvio e ad ogni reload di pagina
  resetAppState();
  window.addEventListener("pageshow", resetAppState);

  // Initialize
  fetchSystemInfo();

  // 1. System Info Check
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/system");
      if (res.ok) {
        const data = await res.json();
        hwStatus.textContent = data.device;
        statDevice.textContent = data.gpu_available ? "GPU (CUDA)" : "CPU";
        if (data.gpu_available) {
          hwPill.style.backgroundColor = "rgba(16, 185, 129, 0.15)";
          hwPill.style.color = "#34d399";
        }
      }
    } catch (err) {
      console.warn("Could not fetch system info:", err);
      hwStatus.textContent = "Server Locale";
    }
  }

  // 2. Drag & Drop and File Upload
  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) processFile(e.target.files[0]);
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  });

  // Paste from clipboard (Ctrl+V)
  window.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let item of items) {
      if (item.type.indexOf("image") !== -1) {
        const blob = item.getAsFile();
        showToast("Immagine incollata dagli appunti!", "success");
        processFile(blob);
        break;
      }
    }
  });

  // Demo Samples Click Handlers
  document.querySelectorAll(".sample-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const type = chip.getAttribute("data-sample");
      generateAndProcessSample(type);
    });
  });

  // 3. Process File via Backend API
  async function processFile(file) {
    if (!file) return;

    // Reset immediato dell'interfaccia prima del nuovo caricamento
    resetAppState();
    showLoading(true, "Caricamento ed elaborazione OCR...", "PaddleOCR sta analizzando il documento");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("lang", langSelect.value);
    formData.append("use_angle_cls", angleClsToggle.checked);
    formData.append("min_confidence", 0.0);

    try {
      const response = await fetch("/api/ocr", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        let errorMsg = `Errore HTTP ${response.status}`;
        try {
          const err = await response.json();
          errorMsg = err.detail || errorMsg;
        } catch (_) {
          try {
            const rawText = await response.text();
            if (rawText && rawText.length < 200) errorMsg = rawText;
          } catch (__) {}
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      currentOcrData = data;
      currentPageIndex = 0;

      // Update statistics
      statBoxes.textContent = data.total_boxes;
      statConf.textContent = `${Math.round(data.avg_confidence * 100)}%`;
      statTime.textContent = `${data.inference_time_ms} ms`;
      statDevice.textContent = data.device;

      // Update PDF pagination toolbar if multiple pages
      if (data.total_pages > 1) {
        pdfNav.classList.remove("hidden");
        updatePageIndicator();
      } else {
        pdfNav.classList.add("hidden");
      }

      // Switch view from dropzone to canvas
      dropzone.classList.add("hidden");
      canvasContainer.classList.remove("hidden");

      // Load page image and render
      renderCurrentPage();
      showToast(`Elaborazione completata in ${data.inference_time_ms} ms!`, "success");

    } catch (err) {
      console.error(err);
      showToast(err.message, "error");
    } finally {
      showLoading(false);
    }
  }

  // 4. Render Current Page & Results
  function renderCurrentPage() {
    if (!currentOcrData || !currentOcrData.pages[currentPageIndex]) return;

    const page = currentOcrData.pages[currentPageIndex];
    
    // Update sidebar text
    fullTextOutput.value = page.full_text || "";
    jsonOutput.textContent = JSON.stringify(currentOcrData, null, 2);

    // Update lines list in sidebar
    renderLinesList(page.lines);

    // Load Image into Canvas
    loadedImage.onload = () => {
      canvas.width = page.width;
      canvas.height = page.height;
      resetZoomFit();
      drawCanvas();
    };
    loadedImage.src = page.image_data;
  }

  function renderLinesList(lines) {
    linesListContainer.innerHTML = "";
    if (!lines || lines.length === 0) {
      linesListContainer.innerHTML = `<div class="empty-state">Nessun testo rilevato in questa pagina.</div>`;
      return;
    }

    lines.forEach((line) => {
      const item = document.createElement("div");
      item.className = "line-item";
      item.id = `line-item-${line.id}`;

      let confClass = "high";
      if (line.confidence < 0.6) confClass = "low";
      else if (line.confidence < 0.85) confClass = "med";

      item.innerHTML = `
        <div class="line-content">
          <span class="line-index">#${line.id + 1}</span>
          <span class="line-text" title="${escapeHtml(line.text)}">${escapeHtml(line.text)}</span>
        </div>
        <span class="confidence-badge ${confClass}">${Math.round(line.confidence * 100)}%</span>
      `;

      // Hover sync with Canvas
      item.addEventListener("mouseenter", () => {
        hoveredBoxId = line.id;
        item.classList.add("active");
        drawCanvas();
      });

      item.addEventListener("mouseleave", () => {
        hoveredBoxId = null;
        item.classList.remove("active");
        drawCanvas();
      });

      item.addEventListener("click", () => {
        selectedBoxId = (selectedBoxId === line.id) ? null : line.id;
        drawCanvas();
      });

      linesListContainer.appendChild(item);
    });
  }

  // 5. Canvas Drawing Engine
  function drawCanvas() {
    if (!loadedImage.complete || !loadedImage.width) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw base image
    ctx.drawImage(loadedImage, 0, 0);

    // 2. Draw OCR bounding boxes if enabled
    if (showBoxes && currentOcrData && currentOcrData.pages[currentPageIndex]) {
      const lines = currentOcrData.pages[currentPageIndex].lines;

      lines.forEach((line) => {
        const isHovered = (hoveredBoxId === line.id);
        const isSelected = (selectedBoxId === line.id);
        const isMatch = searchQuery && line.text.toLowerCase().includes(searchQuery);
        const isCurrentMatch = currentMatchIndex !== -1 && 
          searchResults[currentMatchIndex] && 
          searchResults[currentMatchIndex].pageIndex === currentPageIndex && 
          searchResults[currentMatchIndex].lineId === line.id;

        // Determine color by confidence
        let strokeColor = "#10b981"; // Emerald
        let fillColor = "rgba(16, 185, 129, 0.15)";
        if (line.confidence < 0.6) {
          strokeColor = "#ef4444"; // Rose
          fillColor = "rgba(239, 68, 68, 0.15)";
        } else if (line.confidence < 0.85) {
          strokeColor = "#f59e0b"; // Amber
          fillColor = "rgba(245, 158, 11, 0.15)";
        }

        if (isMatch) {
          strokeColor = "#fbbf24";
          fillColor = "rgba(251, 191, 36, 0.25)";
        }

        if (isCurrentMatch) {
          strokeColor = "#f59e0b";
          fillColor = "rgba(245, 158, 11, 0.45)";
        }

        if (isHovered || isSelected) {
          strokeColor = "#818cf8";
          fillColor = "rgba(99, 102, 241, 0.35)";
        }

        const box = line.box; // [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        ctx.beginPath();
        ctx.moveTo(box[0][0], box[0][1]);
        for (let i = 1; i < box.length; i++) {
          ctx.lineTo(box[i][0], box[i][1]);
        }
        ctx.closePath();

        // Fill and stroke
        ctx.fillStyle = fillColor;
        ctx.fill();

        ctx.lineWidth = (isCurrentMatch) ? 3.5 : (isHovered || isSelected || isMatch) ? 2.5 : 1.8;
        ctx.strokeStyle = strokeColor;
        ctx.stroke();

        // If current match, hovered, or selected, render floating text tag above box
        if (isCurrentMatch || isHovered || isSelected) {
          drawBoxLabel(line, strokeColor, isCurrentMatch ? `🔍 #${currentMatchIndex + 1}` : null);
        }
      });
    }
  }

  function drawBoxLabel(line, color, prefix = null) {
    const box = line.box;
    const x = box[0][0];
    const y = Math.max(16, box[0][1] - 8);
    const label = prefix 
      ? `${prefix}: ${line.text} (${Math.round(line.confidence * 100)}%)`
      : `${line.text} (${Math.round(line.confidence * 100)}%)`;

    ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "rgba(14, 21, 36, 0.9)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x - 4, y - 16, textWidth + 12, 20, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 2, y - 2);
  }

  // 6. Interactive Mouse Events on Canvas (Hover & Click)
  canvas.addEventListener("mousemove", (e) => {
    if (!currentOcrData || !currentOcrData.pages[currentPageIndex]) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);

    const lines = currentOcrData.pages[currentPageIndex].lines;
    let found = null;

    // Check in reverse order so topmost box is picked
    for (let i = lines.length - 1; i >= 0; i--) {
      if (isPointInPolygon([mouseX, mouseY], lines[i].box)) {
        found = lines[i].id;
        break;
      }
    }

    if (found !== hoveredBoxId) {
      hoveredBoxId = found;
      drawCanvas();

      // Sync with sidebar list
      document.querySelectorAll(".line-item").forEach(el => el.classList.remove("active"));
      if (hoveredBoxId !== null) {
        const activeItem = document.getElementById(`line-item-${hoveredBoxId}`);
        if (activeItem) {
          activeItem.classList.add("active");
          activeItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      }
    }
  });

  canvas.addEventListener("mouseleave", () => {
    if (hoveredBoxId !== null) {
      hoveredBoxId = null;
      document.querySelectorAll(".line-item").forEach(el => el.classList.remove("active"));
      drawCanvas();
    }
  });

  // Point in polygon test (Ray-casting algorithm)
  function isPointInPolygon(point, vs) {
    const x = point[0], y = point[1];
    let inside = false;
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
      const xi = vs[i][0], yi = vs[i][1];
      const xj = vs[j][0], yj = vs[j][1];
      const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  // 7. Zoom & Pan Controls
  function updateTransform() {
    canvasWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    zoomDisplay.textContent = `${Math.round(scale * 100)}%`;
  }

  function resetZoomFit() {
    if (!canvas.width || !viewport.clientWidth) return;
    const padding = 40;
    const availW = viewport.clientWidth - padding;
    const availH = viewport.clientHeight - padding;
    const scaleW = availW / canvas.width;
    const scaleH = availH / canvas.height;
    scale = Math.min(scaleW, scaleH, 1.0);
    panX = 0;
    panY = 0;
    updateTransform();
  }

  zoomInBtn.addEventListener("click", () => {
    scale = Math.min(scale * 1.25, 5.0);
    updateTransform();
  });

  zoomOutBtn.addEventListener("click", () => {
    scale = Math.max(scale / 1.25, 0.15);
    updateTransform();
  });

  zoomFitBtn.addEventListener("click", resetZoomFit);

  // Pan interaction on viewport
  viewport.addEventListener("mousedown", (e) => {
    // Only pan on middle click or left click on viewport background
    if (e.button === 1 || e.target === viewport) {
      isPanning = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
      viewport.classList.add("grabbing");
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    updateTransform();
  });

  window.addEventListener("mouseup", () => {
    if (isPanning) {
      isPanning = false;
      viewport.classList.remove("grabbing");
    }
  });

  // Wheel zoom
  viewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    scale = Math.min(Math.max(scale * zoomFactor, 0.15), 6.0);
    updateTransform();
  }, { passive: false });

  // Toggle Box Button
  toggleBoxesBtn.addEventListener("click", () => {
    showBoxes = !showBoxes;
    toggleBoxesBtn.classList.toggle("active", showBoxes);
    drawCanvas();
  });

  // Clear / New Document Button
  clearBtn.addEventListener("click", resetAppState);

  // 8. PDF Page Navigation
  prevPageBtn.addEventListener("click", () => {
    if (currentPageIndex > 0) {
      currentPageIndex--;
      updatePageIndicator();
      renderCurrentPage();
    }
  });

  nextPageBtn.addEventListener("click", () => {
    if (currentOcrData && currentPageIndex < currentOcrData.total_pages - 1) {
      currentPageIndex++;
      updatePageIndicator();
      renderCurrentPage();
    }
  });

  function updatePageIndicator() {
    pageIndicator.textContent = `Pagina ${currentPageIndex + 1} / ${currentOcrData.total_pages}`;
  }

  // 9. Tabs Navigation
  function activateTab(tabId) {
    tabBtns.forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
    });
    tabPanes.forEach(p => {
      p.classList.toggle("active", p.id === tabId);
    });
  }

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      activateTab(btn.getAttribute("data-tab"));
    });
  });

  // 10. Multi-Page Global Search Engine
  function performGlobalSearch() {
    searchQuery = searchInput.value.trim().toLowerCase();
    clearSearchBtn.classList.toggle("hidden", searchQuery === "");

    searchResults = [];
    currentMatchIndex = -1;

    if (!currentOcrData || !searchQuery) {
      searchMatchCount.classList.add("hidden");
      searchNavBtns.classList.add("hidden");
      searchBadgeCount.classList.add("hidden");
      searchResultsContainer.innerHTML = `<div class="empty-state">Digita un termine nella barra di ricerca per visualizzare i risultati in tutte le pagine.</div>`;
      document.querySelectorAll(".line-item").forEach(item => item.classList.remove("matched"));
      drawCanvas();
      return;
    }

    // Scan all pages and collect occurrences
    currentOcrData.pages.forEach((page, pIdx) => {
      page.lines.forEach((line) => {
        if (line.text.toLowerCase().includes(searchQuery)) {
          searchResults.push({
            globalIndex: searchResults.length,
            pageIndex: pIdx,
            pageNumber: page.page_number,
            lineId: line.id,
            text: line.text,
            confidence: line.confidence,
            box: line.box
          });
        }
      });
    });

    // Update match count badge on tab
    searchBadgeCount.textContent = searchResults.length;
    searchBadgeCount.classList.toggle("hidden", searchResults.length === 0);

    if (searchResults.length === 0) {
      searchMatchCount.textContent = "0 trovati";
      searchMatchCount.classList.remove("hidden");
      searchNavBtns.classList.add("hidden");
      searchResultsContainer.innerHTML = `<div class="empty-state">Nessuna corrispondenza trovata per "<strong>${escapeHtml(searchQuery)}</strong>".</div>`;
      document.querySelectorAll(".line-item").forEach(item => item.classList.remove("matched"));
      drawCanvas();
      return;
    }

    // Results found: render grouped by page in the search results tab
    searchNavBtns.classList.remove("hidden");
    searchMatchCount.classList.remove("hidden");

    // Automatically switch to Search Tab so user immediately sees results
    activateTab("searchTab");

    renderSearchResultsList();

    // Focus on first match
    jumpToMatch(0);
  }

  function renderSearchResultsList() {
    searchResultsContainer.innerHTML = "";

    const grouped = {};
    searchResults.forEach(match => {
      if (!grouped[match.pageIndex]) grouped[match.pageIndex] = [];
      grouped[match.pageIndex].push(match);
    });

    Object.keys(grouped).forEach(pageIdxStr => {
      const pIdx = parseInt(pageIdxStr);
      const matchesOnPage = grouped[pIdx];
      const pageNum = matchesOnPage[0].pageNumber;

      const groupDiv = document.createElement("div");
      groupDiv.className = "search-page-group";

      const header = document.createElement("div");
      header.className = "search-page-header";
      header.innerHTML = `
        <span>📄 Pagina ${pageNum}</span>
        <span>${matchesOnPage.length} ${matchesOnPage.length === 1 ? 'risultato' : 'risultati'}</span>
      `;
      groupDiv.appendChild(header);

      matchesOnPage.forEach(match => {
        const card = document.createElement("div");
        card.className = "search-result-card";
        card.id = `search-match-card-${match.globalIndex}`;

        const highlightedText = highlightQueryInText(match.text, searchQuery);

        let confClass = "high";
        if (match.confidence < 0.6) confClass = "low";
        else if (match.confidence < 0.85) confClass = "med";

        card.innerHTML = `
          <div class="search-card-meta">
            <span>Riga #${match.lineId + 1}</span>
            <span class="confidence-badge ${confClass}">${Math.round(match.confidence * 100)}%</span>
          </div>
          <div class="search-card-text">${highlightedText}</div>
        `;

        card.addEventListener("click", () => {
          jumpToMatch(match.globalIndex);
        });

        groupDiv.appendChild(card);
      });

      searchResultsContainer.appendChild(groupDiv);
    });
  }

  function highlightQueryInText(text, query) {
    if (!query) return escapeHtml(text);
    const regex = new RegExp(`(${escapeRegex(query)})`, "gi");
    return escapeHtml(text).replace(regex, '<mark class="search-highlight">$1</mark>');
  }

  function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function jumpToMatch(index) {
    if (!searchResults.length) return;
    if (index < 0) index = searchResults.length - 1;
    if (index >= searchResults.length) index = 0;

    currentMatchIndex = index;
    const match = searchResults[currentMatchIndex];

    // Update match count display: e.g. "1 / 4 (Pag. 1)"
    searchMatchCount.textContent = `${currentMatchIndex + 1} / ${searchResults.length} (Pag. ${match.pageNumber})`;

    // If match is on a different page, change page!
    if (match.pageIndex !== currentPageIndex) {
      currentPageIndex = match.pageIndex;
      updatePageIndicator();
      renderCurrentPage();
    } else {
      // Highlight matching items in current linesTab
      document.querySelectorAll(".line-item").forEach(item => {
        const text = item.querySelector(".line-text").textContent.toLowerCase();
        item.classList.toggle("matched", text.includes(searchQuery));
      });
    }

    // Highlight active card in search tab
    document.querySelectorAll(".search-result-card").forEach(c => c.classList.remove("active-match"));
    const activeCard = document.getElementById(`search-match-card-${match.globalIndex}`);
    if (activeCard) {
      activeCard.classList.add("active-match");
      activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Focus on canvas
    selectedBoxId = match.lineId;
    hoveredBoxId = match.lineId;
    drawCanvas();
  }

  searchInput.addEventListener("input", performGlobalSearch);

  searchNextBtn.addEventListener("click", () => jumpToMatch(currentMatchIndex + 1));
  searchPrevBtn.addEventListener("click", () => jumpToMatch(currentMatchIndex - 1));

  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (e.shiftKey) {
        jumpToMatch(currentMatchIndex - 1);
      } else {
        jumpToMatch(currentMatchIndex + 1);
      }
    } else if (e.key === "Escape") {
      searchInput.value = "";
      performGlobalSearch();
    }
  });

  clearSearchBtn.addEventListener("click", () => {
    searchInput.value = "";
    performGlobalSearch();
  });

  // 11. Actions & Export
  copyTextBtn.addEventListener("click", () => {
    const text = fullTextOutput.value;
    if (!text) {
      showToast("Nessun testo da copiare!", "error");
      return;
    }
    navigator.clipboard.writeText(text).then(() => {
      showToast("Testo copiato negli appunti!", "success");
    }).catch(err => {
      showToast("Impossibile copiare il testo: " + err, "error");
    });
  });

  // Dropdown toggle
  exportDropdownBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    exportMenu.classList.toggle("show");
  });

  // Stampa nativa del documento originale a piena risoluzione
  function triggerPrintOriginal() {
    if (!currentOcrData || !currentOcrData.doc_id) {
      showToast("Nessun documento attivo da stampare!", "error");
      return;
    }
    const docId = currentOcrData.doc_id;
    const printUrl = `/api/document/${docId}/original`;

    showToast("Apertura finestra di stampa con PDF originale...", "info");
    const printWindow = window.open(printUrl, "_blank");
    if (printWindow) {
      printWindow.addEventListener("load", () => {
        try {
          printWindow.print();
        } catch (_) {}
      });
    } else {
      showToast("Finestra popup bloccata dal browser. Consenti i popup per stampare.", "error");
    }
  }

  if (printDocBtn) printDocBtn.addEventListener("click", triggerPrintOriginal);
  if (printDropdownBtn) printDropdownBtn.addEventListener("click", triggerPrintOriginal);

  async function exportPdf(mode) {
    if (!currentOcrData) {
      showToast("Nessun dato OCR da esportare!", "error");
      return;
    }
    showLoading(true, "Generazione PDF in corso...", mode === "searchable" ? "Composizione PDF ricercabile con struttura originale" : "Creazione PDF vettoriale con layout fedele");
    try {
      const response = await fetch("/api/export/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: mode,
          ocr_data: currentOcrData,
          doc_id: currentOcrData ? currentOcrData.doc_id : null
        })
      });
      if (!response.ok) {
        throw new Error("Errore durante la generazione del PDF");
      }
      const blob = await response.blob();
      const filename = `paddleocr_${mode}_${Date.now()}.pdf`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast(`PDF (${mode === "searchable" ? "Ricercabile" : "Ricostruito"}) scaricato con successo!`, "success");
    } catch (err) {
      console.error(err);
      showToast(err.message, "error");
    } finally {
      showLoading(false);
    }
  }

  exportPdfSearchableBtn.addEventListener("click", () => exportPdf("searchable"));
  exportPdfReconstructedBtn.addEventListener("click", () => exportPdf("reconstructed"));

  exportTxtBtn.addEventListener("click", () => {
    if (!currentOcrData) return;
    const text = currentOcrData.pages.map(p => `--- Pagina ${p.page_number} ---\n` + p.full_text).join("\n\n");
    downloadFile(text, "paddleocr_estratto.txt", "text/plain");
  });

  exportJsonBtn.addEventListener("click", () => {
    if (!currentOcrData) return;
    downloadFile(JSON.stringify(currentOcrData, null, 2), "paddleocr_dati.json", "application/json");
  });

  exportCsvBtn.addEventListener("click", () => {
    if (!currentOcrData) return;
    let csv = "Pagina,ID,Testo,Confidenza,Coordinate\n";
    currentOcrData.pages.forEach(p => {
      p.lines.forEach(l => {
        const cleanText = `"${l.text.replace(/"/g, '""')}"`;
        const coords = `"${JSON.stringify(l.box)}"`;
        csv += `${p.page_number},${l.id + 1},${cleanText},${l.confidence},${coords}\n`;
      });
    });
    downloadFile(csv, "paddleocr_righe.csv", "text/csv");
  });

  function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type: `${type};charset=utf-8;` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`File ${filename} scaricato!`, "success");
  }

  // 12. Synthetic Sample Generator (Instant Testing)
  function generateAndProcessSample(type) {
    const sampleCanvas = document.createElement("canvas");
    const sCtx = sampleCanvas.getContext("2d");

    if (type === "receipt") {
      sampleCanvas.width = 600;
      sampleCanvas.height = 800;
      sCtx.fillStyle = "#ffffff";
      sCtx.fillRect(0, 0, 600, 800);

      sCtx.fillStyle = "#1e293b";
      sCtx.font = "bold 28px sans-serif";
      sCtx.fillText("SUPERMERCATO ITALIA", 120, 80);

      sCtx.font = "16px sans-serif";
      sCtx.fillText("Via Roma 145, 00100 Roma (RM)", 160, 120);
      sCtx.fillText("P.IVA 01234567890 - Tel: 06 1234567", 140, 145);
      sCtx.fillText("--------------------------------------------------", 60, 180);

      sCtx.font = "bold 18px monospace";
      sCtx.fillText("DESCRIZIONE ARTICOLO       QTA    PREZZO", 60, 220);
      sCtx.font = "16px monospace";
      sCtx.fillText("PASTA BARILLA 500G          2      2.40 EUR", 60, 260);
      sCtx.fillText("LATTE PARMALAT 1L           1      1.50 EUR", 60, 300);
      sCtx.fillText("CAFFE LAVAZZA 250G          1      3.80 EUR", 60, 340);
      sCtx.fillText("PANE FRESCO 1KG             1      2.20 EUR", 60, 380);
      sCtx.fillText("--------------------------------------------------", 60, 420);

      sCtx.font = "bold 22px monospace";
      sCtx.fillText("TOTALE COMPLESSIVO:                9.90 EUR", 60, 470);
      sCtx.font = "16px sans-serif";
      sCtx.fillText("PAGAMENTO: CARTA DI CREDITO", 60, 520);
      sCtx.fillText("GRAZIE E ARRIVEDERCI!", 180, 600);

    } else if (type === "invoice") {
      sampleCanvas.width = 750;
      sampleCanvas.height = 900;
      sCtx.fillStyle = "#ffffff";
      sCtx.fillRect(0, 0, 750, 900);

      sCtx.fillStyle = "#0f172a";
      sCtx.font = "bold 32px sans-serif";
      sCtx.fillText("FATTURA COMMERCIALE", 50, 80);

      sCtx.font = "bold 16px sans-serif";
      sCtx.fillText("Fattura N: 2026/0452", 50, 130);
      sCtx.fillText("Data: 22 Settembre 2026", 50, 160);

      sCtx.font = "15px sans-serif";
      sCtx.fillText("Destinatario: Mario Rossi", 420, 130);
      sCtx.fillText("Codice Fiscale: RSSMRA80A01H501U", 420, 160);
      sCtx.fillText("Indirizzo: Corso Vittorio Emanuele 12", 420, 190);

      sCtx.fillStyle = "#e2e8f0";
      sCtx.fillRect(50, 240, 650, 35);
      sCtx.fillStyle = "#0f172a";
      sCtx.font = "bold 15px sans-serif";
      sCtx.fillText("Descrizione Servizio", 60, 263);
      sCtx.fillText("Importo", 600, 263);

      sCtx.font = "15px sans-serif";
      sCtx.fillText("Consulenza Sviluppo Software e Intelligenza Artificiale", 60, 320);
      sCtx.fillText("1.250,00 EUR", 600, 320);

      sCtx.fillText("Configurazione Server PaddleOCR su GPU RTX", 60, 370);
      sCtx.fillText("600,00 EUR", 600, 370);

      sCtx.fillStyle = "#0f172a";
      sCtx.fillRect(50, 420, 650, 2);

      sCtx.font = "bold 18px sans-serif";
      sCtx.fillText("TOTALE FATTURA:", 400, 470);
      sCtx.fillText("1.850,00 EUR", 600, 470);

    } else { // signboard
      sampleCanvas.width = 700;
      sampleCanvas.height = 450;
      sCtx.fillStyle = "#1e3a8a";
      sCtx.fillRect(0, 0, 700, 450);

      sCtx.strokeStyle = "#ffffff";
      sCtx.lineWidth = 10;
      sCtx.strokeRect(20, 20, 660, 410);

      sCtx.fillStyle = "#ffffff";
      sCtx.font = "bold 52px sans-serif";
      sCtx.fillText("CENTRO STORICO", 110, 180);

      sCtx.font = "bold 32px sans-serif";
      sCtx.fillText("ZONA A TRAFFICO LIMITATO", 110, 250);

      sCtx.fillStyle = "#facc15";
      sCtx.font = "bold 26px sans-serif";
      sCtx.fillText("VARCO ATTIVO H24", 210, 330);
    }

    sampleCanvas.toBlob((blob) => {
      const sampleFile = new File([blob], `sample_${type}.png`, { type: "image/png" });
      processFile(sampleFile);
    }, "image/png");
  }

  // 13. Helpers
  function showLoading(show, title = "", subtitle = "") {
    if (show) {
      if (title) loadingText.textContent = title;
      loadingOverlay.classList.remove("hidden");
    } else {
      loadingOverlay.classList.add("hidden");
    }
  }

  function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span>${type === "success" ? "✓" : "⚠"}</span>
      <span>${escapeHtml(message)}</span>
    `;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
