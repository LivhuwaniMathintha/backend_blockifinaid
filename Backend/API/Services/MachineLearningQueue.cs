
using MassTransit;


namespace API.Services
{
    public class MLQueuePublisher
    {
        private readonly IPublishEndpoint _publishEndpoint;
        public MLQueuePublisher(IPublishEndpoint publishEndpoint)
        {
            _publishEndpoint = publishEndpoint;
        }

        public async Task  PublishPaymentCreated(MlFraudDetectionEvent fraudDetectionCreatedEvent)
        {
            if (fraudDetectionCreatedEvent == null)
            {
                throw new ArgumentNullException(nameof(fraudDetectionCreatedEvent), "PaymentCreatedEvent cannot be null");
            }

            await _publishEndpoint.Publish(fraudDetectionCreatedEvent);

        }

    }

    public class MlFraudDetectionEvent
    {
        public int Amount { get; set; }
        public string Funder { get; set; } = string.Empty;
        public string PaymentType { get; set; } = string.Empty;
        public int Month { get; set; }
        public string StudentId { get; set; } = string.Empty;
        public int DuplicateCount { get; set; }
        public int ActivationMonth { get; set; }

    }
    public class MLFraudDetectionEventModel : MlFraudDetectionEvent
    {
        public int Id { get; set; }
    }

    public class MLQueueConsumer : IConsumer<MlFraudDetectionEvent>
    {

        private readonly ILogger<FinAidQueueConsumer> _logger;
        private readonly IEmailService _emailService;
        public bool Is_Paid { get; set; } = false;

        public MLQueueConsumer(ILogger<FinAidQueueConsumer> logger, IEmailService emailService)
        {
            _logger = logger;
            _emailService = emailService;
        }

        public async Task Consume(ConsumeContext<MlFraudDetectionEvent> context)
        {
            var fraudDetectionEvent = new MlFraudDetectionEvent
            {
                Amount = context.Message.Amount,
                Funder = context.Message.PaymentType,
                PaymentType = context.Message.PaymentType,
                Month = context.Message.Month,
                StudentId = context.Message.StudentId,
                DuplicateCount = context.Message.DuplicateCount,
                ActivationMonth = DateTime.UtcNow.Month
            };

            if (fraudDetectionEvent is null)
            {
                _logger.LogError("PaymentCreatedEvent is null, cannot publish to queue.");
            }
            else
            {
             
            }
        }
    }
}