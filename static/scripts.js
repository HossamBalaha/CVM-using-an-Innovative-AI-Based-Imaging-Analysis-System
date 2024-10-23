function ResetProgress() {
  $("#progress").addClass("d-none");
  $("#progressText").text("0");
  $("#progressBar").css("width", "0%");
  $("form :input").prop("disabled", false);
  $("form button").prop("disabled", false);
}


function UpdateProgress(progress) {
  progress = Math.round(progress * 100) / 100.0;
  $("#progress").removeClass("d-none");
  $("#progressText").text(progress);
  $("#progressBar").css("width", progress + "%");
  $("form :input").prop("disabled", true);
  $("form button").prop("disabled", true);
}

function CheckProgress(url) {
  if (url == null) {
    return;
  }

  $.ajax({
    url: url, method: 'GET', success: function (response) {
      const progress = response.progress;
      if (progress >= 100) {
        ResetProgress();
      } else if (progress <= 0) {
        ResetProgress();
        // setTimeout(CheckProgress, 1000, url);
      } else {
        UpdateProgress(response.progress);
        setTimeout(CheckProgress, 10000, url);
      }
    }, error: function (error) {
      console.log(error);
    }
  });
}

new ShowMore('.showMore');
