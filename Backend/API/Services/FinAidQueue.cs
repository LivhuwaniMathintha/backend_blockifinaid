
using API.Data.Queue;
using MassTransit;


namespace API.Services
{
    public class FinAidQueuePublisher
    {
        private readonly IPublishEndpoint _publishEndpoint;

        public FinAidQueuePublisher(IPublishEndpoint publishEndpoint)
        {
            _publishEndpoint = publishEndpoint;
        }

        public async Task  PublishPaymentCreated(PaymentCreatedEvent paymentCreatedEvent)
        {
            if (paymentCreatedEvent == null)
            {
                throw new ArgumentNullException(nameof(paymentCreatedEvent), "PaymentCreatedEvent cannot be null");
            }

            await _publishEndpoint.Publish(paymentCreatedEvent);

        }

    }

    public class FinAidQueueConsumer : IConsumer<PaymentCreatedEvent>{

        private readonly ILogger<FinAidQueueConsumer> _logger;
        private readonly IEmailService _emailService;
        public bool Is_Paid { get; set; } = false;

        public FinAidQueueConsumer(ILogger<FinAidQueueConsumer> logger, IEmailService emailService)
        {
            _logger = logger;
            _emailService = emailService;
        }

        public async Task Consume(ConsumeContext<PaymentCreatedEvent> context)
        {

            var payment = new PaymentCreatedEvent(
                context.Message.PaymentId,
                context.Message.StuduentEmail,
                context.Message.StudentName,
                context.Message.UserId,
                context.Message.StudnetBankAccountNumber,
                context.Message.Amount,
                context.Message.PaymentType,
                context.Message.StudentBankName,
                context.Message.StudentBankBranchCode,
                context.Message.OrgBankAccountNumber,
                context.Message.OrgBankName,
                context.Message.OrgBankBranchCode,
                context.Message.CreatedAt,
                context.Message.IsPaid
            );

            if (payment is null)
            {
                _logger.LogError("PaymentCreatedEvent is null, cannot publish to queue.");
            }
            else
            {
                // Process the payment event here


                // send confirmation message to the student here

                Is_Paid = true;

                await _emailService.SendEmailAsync(payment.StuduentEmail,
                    "Payment Confirmation",
                    $"""
                        <p>Dear {payment.StudentName},</p><br><p>Your payment of {payment.Amount:C} has been successfully processed and will reflect within 48 hours</p><p><p>Best regards,</p><p>Financial Aid Team</p> 
                    """,
                    true
                );
            }
        }
    }
}