let totalReviews = 0
let aiReviews = 0
let humanReviews = 0

function showPage(page){

document.querySelectorAll(".page").forEach(p=>{
p.classList.remove("active")
})

document.getElementById(page).classList.add("active")

}

async function detectReview(){

const review_text = document.getElementById("review_text").value.trim()
const overall = Number(document.getElementById("overall").value)
const helpful_ratio = Number(document.getElementById("helpful_ratio").value)

const result = document.getElementById("result")

if(!review_text){
alert("Enter review text")
return
}

result.style.display="block"
result.innerHTML="Analyzing..."

try{

const response = await fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        review_text,
        overall,
        helpful_ratio
    })
});

const data = await response.json();

if(data.error){
    result.innerHTML = "Server Error: " + data.error;
    return;
}

result.innerHTML = `
Prediction: <b>${data.prediction}</b><br><br>
Rating: ${data.rating_value}/5<br>
Quality: ${data.rating_description}<br><br>
Helpfulness: ${data.helpful_description}
`;

totalReviews++

if(data.prediction=="AI-generated"){
aiReviews++
result.className="ai"
}else{
humanReviews++
result.className="human"
}

document.getElementById("totalReviews").innerText = totalReviews
document.getElementById("aiReviews").innerText = aiReviews
document.getElementById("humanReviews").innerText = humanReviews

result.innerHTML = `
Prediction: ${data.prediction} <br><br>
Rating: ${data.rating_value}/5 <br>
Quality: ${data.rating_description} <br><br>
Helpfulness: ${data.helpful_description}
`

}catch{

result.innerHTML="good product"

}

}