using System.Text;
using System.Threading.Tasks;
using API.Data.Queue;
using API.Services;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
namespace API.Controllers
{
    [ApiController]
    [Route("api/v1/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly ILogger<AuthController> _logger;
        private readonly IEmailService _emailService;
        private readonly FinAidQueuePublisher _finAidQueuePublisher;
        private readonly HttpClient _httpClient;

        public AuthController(ILogger<AuthController> logger, IEmailService emailService, FinAidQueuePublisher finAidQueuePublisher, HttpClient httpClient)
        {
            _logger = logger;
            _emailService = emailService;
            _finAidQueuePublisher = finAidQueuePublisher;
            _httpClient = httpClient;
            _httpClient.BaseAddress = new Uri("http://127.0.0.1:8000/"); // Set your base address for the HTTP client
        }

        [HttpGet]
        public async Task<IActionResult> GetBasicData()
        {
            _logger.LogInformation("This is a basic endpoint for the Financial Aid API in the Auth Endpoint");

            // create a payment created event example
            var paymentCreatedEvent = new PaymentCreatedEvent(
                PaymentId: "12345",
                StuduentEmail: "zitoesn@gmail.com",
                StudentName: "Zito Esn",
                UserId: "user-123",
                StudnetBankAccountNumber: "1234567890",
                Amount: 1000.00m,
                PaymentType: "Scholarship",
                StudentBankName: "Bank of Example",
                StudentBankBranchCode: "001",
                OrgBankAccountNumber: "0987654321",
                OrgBankName: "Organization Bank",
                OrgBankBranchCode: "002",
                CreatedAt: DateTime.UtcNow
            );
            // Publish the payment created event

            // Make an API call to the ml/payments/fraud_detection:
            var fraud_detection_payload = new
            {
                amount = 1650,
                funder = "NSFAS",
                payment_type = "food",
                month = 2,
                student_id = "S000441",
                duplicate_count = 1,
                activation_month = 1
            };

            var is_fraud_response = await _httpClient.PostAsync("ml/payments/fraud_detection", new StringContent(JsonConvert.SerializeObject(fraud_detection_payload), Encoding.UTF8, "application/json"));

            if (is_fraud_response.IsSuccessStatusCode)
            {
                var responseContent = await is_fraud_response.Content.ReadAsStringAsync();
                _logger.LogInformation("Fraud Detection Response: {Response}", responseContent);

                //check if responseContent.is_fraud is true or false
                var fraudDetectionResult = JsonConvert.DeserializeObject<dynamic>(responseContent);
                if (fraudDetectionResult?.is_fraud == false)
                {
                    try
                    {
                        await _finAidQueuePublisher.PublishPaymentCreated(paymentCreatedEvent);
                        _logger.LogInformation("Payment Created Event published successfully");
                        return Ok(new { message = $"Payment Created Event published successfully with no fraud detected at a risk score of {fraudDetectionResult?.risk_score}", paymentId = paymentCreatedEvent.PaymentId });
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error publishing Payment Created Event");
                    }

                }
                else
                {
                    _logger.LogError("Failed to call fraud detection service. Status Code: {StatusCode}", is_fraud_response.StatusCode);
                    return BadRequest(new { message = "Fraud detected, payment not processed.", riskScore = fraudDetectionResult?.risk_score });
                }

            }
          
            return BadRequest(new { message = "Failed to call fraud detection service.", statusCode = is_fraud_response.StatusCode });
      
        }
    }
}