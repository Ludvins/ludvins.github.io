(function () {
  var root = document.documentElement;
  var themeToggle = document.querySelector("#theme-toggle");
  var themeLabel = document.querySelector("[data-theme-label]");
  var themeColor = document.querySelector("#theme-color");
  var systemTheme = window.matchMedia("(prefers-color-scheme: dark)");

  function savedTheme() {
    try {
      return localStorage.getItem("site-theme");
    } catch (error) {
      return null;
    }
  }

  function applyTheme(theme, persist) {
    var isDark = theme === "dark";
    root.dataset.theme = isDark ? "dark" : "light";

    if (themeToggle) {
      themeToggle.setAttribute("aria-pressed", String(isDark));
      themeToggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    }
    if (themeLabel) {
      themeLabel.textContent = isDark ? "Light" : "Dark";
    }
    if (themeColor) {
      themeColor.content = isDark ? "#171613" : "#f4efe6";
    }
    if (persist) {
      try {
        localStorage.setItem("site-theme", isDark ? "dark" : "light");
      } catch (error) {
        // The selected theme still applies for the current page.
      }
    }
  }

  applyTheme(root.dataset.theme || (systemTheme.matches ? "dark" : "light"), false);

  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
    });
  }

  if (systemTheme.addEventListener) {
    systemTheme.addEventListener("change", function (event) {
      if (!savedTheme()) {
        applyTheme(event.matches ? "dark" : "light", false);
      }
    });
  }

  var list = document.querySelector("#publication-list");
  if (!list) {
    return;
  }

  var search = document.querySelector("#publication-search");
  var chips = Array.prototype.slice.call(document.querySelectorAll("[data-filter]"));
  var yearButtons = Array.prototype.slice.call(document.querySelectorAll("[data-year]"));
  var cards = Array.prototype.slice.call(list.querySelectorAll(".publication-card"));
  var noResults = list.querySelector(".no-results");
  var activeFilter = "all";
  var activeYear = "all";

  function normalize(value) {
    return (value || "").toLowerCase().trim();
  }

  function setActive(items, activeItem) {
    items.forEach(function (item) {
      item.classList.toggle("is-active", item === activeItem);
    });
  }

  function applyFilters() {
    var query = normalize(search && search.value);
    var visibleCount = 0;

    cards.forEach(function (card) {
      var haystack = normalize(card.dataset.title + " " + card.textContent);
      var tags = normalize(card.dataset.tags);
      var year = normalize(card.dataset.year);
      var matchesSearch = !query || haystack.indexOf(query) !== -1;
      var matchesFilter = activeFilter === "all" || tags.indexOf(activeFilter) !== -1;
      var matchesYear = activeYear === "all" || year === activeYear;
      var isVisible = matchesSearch && matchesFilter && matchesYear;

      card.hidden = !isVisible;
      if (isVisible) {
        visibleCount += 1;
      }
    });

    if (noResults) {
      noResults.hidden = visibleCount !== 0;
    }
  }

  if (search) {
    search.addEventListener("input", applyFilters);
  }

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      activeFilter = chip.dataset.filter || "all";
      setActive(chips, chip);
      applyFilters();
    });
  });

  yearButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      activeYear = button.dataset.year || "all";
      setActive(yearButtons, button);
      applyFilters();
    });
  });
})();
